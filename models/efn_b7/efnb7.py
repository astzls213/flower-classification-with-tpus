# 请在kaggle notebook上运行，如果用gpu，请自己前往
# tensorflow官网，查询gpu配置过程

!pip install -q efficientnet
#导入核心库
import tensorflow as tf # gg
import efficientnet.tfkeras as efn # 核心网络
import numpy as np # 矩阵处理
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
# 高级统计数据
import matplotlib.pyplot as plt # 画图
import re # 统计图片数量
import json # 保存一些信息
from math import cos,pi,e # 构造学习率调度函数
from time import perf_counter # 用于计算时间
print('Tensorflow ver:',tf.__version__) # 导入完成！

try:
    #解析TPU芯片集群
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    #kaggle不需要喂参数，他默认好了
    tf.config.experimental_connect_to_cluster(tpu)
    # 配置运算器
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # 初始化tpu系统
    print('TPU configured, at',tpu.master())
except ValueError: #本地运行，CPU
    tpu = None
    print('TPUs not found! Using CPU')

#获得分配策略，进行并行运算，以提升训练速度
if tpu:
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print('Succeed distributing TPUs!')
print("REPLICAS: ", strategy.num_replicas_in_sync)

#后面很多核心函数都有优化参数，根据性能自动选择
AUTO = tf.data.experimental.AUTOTUNE
print('AUTO:',AUTO)

#超参数定义
#strategy来自tpu_settings.py
BATCH_SIZE = 16 * strategy.num_replicas_in_sync # 根据cpu/tpu自动调整batch大小
EPOCHS = 12 # 训练周次


IMAGE_SIZE = [512,512] #手动修改此处图像大小，进行训练
WIDTH = IMAGE_SIZE[0]
HEIGHT = IMAGE_SIZE[1]
CHANNELS = 3


# 加载官方数据集
try: #Running in Kaggle kernel
    from kaggle_datasets import KaggleDatasets
    BASE = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')
except ModuleNotFoundError: # Running at my mac
    BASE = "/Users/astzls/Downloads/flower"

PATH_SELECT = { # 根据图像大小选择路径
    192: BASE + '/tfrecords-jpeg-192x192',
    224: BASE + '/tfrecords-jpeg-224x224',
    331: BASE + '/tfrecords-jpeg-331x331',
    512: BASE + '/tfrecords-jpeg-512x512'
}
IMAGE_PATH = PATH_SELECT[IMAGE_SIZE[0]]

#此处利用tf.io的库函数
#读出文件集方式很多种，也可以用os+re库进行
TRAINING_FILENAMES = tf.io.gfile.glob(IMAGE_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(IMAGE_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(IMAGE_PATH + '/test/*.tfrec')

# 牛津大学花蕊数据（原：7310 数据清洗后：2718）Oxford 102 for TPU competition
# 使用后，精确度提升2%
try: #Running in Kaggle kernel
    from kaggle_datasets import KaggleDatasets
    ext = KaggleDatasets().get_gcs_path('oxford-102-for-tpu-competition')
except ModuleNotFoundError: # Running at my mac
    ext = None

ext_path_select = { # 根据图像大小选择路径
    192: ext + '/tfrecords-jpeg-192x192',
    224: ext + '/tfrecords-jpeg-224x224',
    331: ext + '/tfrecords-jpeg-331x331',
    512: ext + '/tfrecords-jpeg-512x512'
}
ext_path = ext_path_select[IMAGE_SIZE[0]]

TRAINING_FILENAMES += tf.io.gfile.glob(ext_path + '/*.tfrec')


# 统计image数量
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

#下面是核心函数构造，自顶向下
def get_training_dataset():
    #严格遵循代码开头注释流程
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True) #labeled参数是用来判断数据有没有label的，比如test dataset就没有label
    #装载分离好等数据，每个example含image，class等等，其中kaggle说image的图像是jpeg格式的
    dataset = dataset.map(data_augmentation, num_parallel_calls=AUTO)
    #进行数据扩容，此步非常重要！可大大提升训练精度，已被广泛使用，不用你就out了
    dataset = dataset.shuffle(2048)
    # 每次取2048个examples，打乱。这个值不用过大也不要过小
    dataset = dataset.batch(BATCH_SIZE)
    #批处理
    dataset = dataset.repeat()
    #repeat无参数说明无限重复dataset。放心，不会内存溢出，只是懒循环罢了
    dataset = dataset.prefetch(AUTO)
    #根据cpu决定是否提前准备数据。为什么要这么做？原因是我想采用tpu进行训练，那么在tpu在训练时，cpu可以预先把下一批图像load到内存，当
    # tpu训练好了，直接又能进行下一批当训练，减少了训练时间。
    # 还是那句话，不用，你就out了 
    return dataset

def get_validation_dataset(order=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True,ordered=order)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache() #cache dataset到memory，加速训练时间，只有3712张
    dataset = dataset.prefetch(AUTO) 
    return dataset

def get_test_dataset(order=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False,ordered=order)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
    return dataset

#完善核心函数中的小函数
def load_dataset(filenames, labeled=True, ordered=False):
    #ordered参数是指并行读取数据时不必在意是否按照原来顺序，加快速度
    #顺序不重要的= =

    #不在意顺序预处理
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False #详见help

    #利用API导入初始TFrec文件集数据
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    #设置dataset，让他保持并行读出来的顺序就行了
    dataset = dataset.with_options(ignore_order)
    #根据label决定传入带标签的解析函数，还是不带标签（test只有id）的解析函数
    dataset = dataset.map(read_labeled_tfrecord if labeled
                     else read_unlabeled_tfrecord, 
                     num_parallel_calls=AUTO)
    return dataset #现在dataset的每个data有两个部分了，一个是image，一个是class或id号

def data_augmentation(image, label):
    #所谓数据扩容，就是把原来的照片左移或右移，或上下左右反转一下，就得到了“新”图像
    #此方法利用了现实世界的平移不变形和空间不变形

    image = tf.image.random_flip_left_right(image) 
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)

    # 还有下列api进行更无聊的处理
    # 'random_contrast', 对比度
    # 'random_crop', 缩放
    # 'random_hue', 色相
    # 'random_jpeg_quality',  图片质量
    # 'random_saturation' 饱和度
    return image, label   

def decode_image(image_data):
    #由于给的图像是jpeg格式，故用对应api进行处理。
    #为什么要decode，因为他把图片写成bytes串了
    image = tf.image.decode_jpeg(image_data, channels=3)
    #得到tf.Tensor形式的image
    image = tf.cast(image, tf.float32)
    image /= 255.0
    #将image的每个数值调整在[0,1]之间，方便训练
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    #reshape此部非常重要，调试的时候被坑了，老是说什么shape不匹配
    return image

def read_labeled_tfrecord(example):
    FEATURE = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64)  
    }
    example = tf.io.parse_single_example(example, FEATURE)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label #返回一个以 图像数组和标签形式的数据集

def read_unlabeled_tfrecord(example):
    FEATURE = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, FEATURE)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def show_acc_loss(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.yticks([0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Efficient net b7')
    plt.legend()
    plt.savefig('acc.jpg')

    plt.figure()

    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss & CE')
    plt.title('Efficient net b7')
    plt.legend()

    plt.savefig('loss.jpg')

    plt.show()
    
# 创建efnb7 & custom classifier
with strategy.scope():
    conv_base = efn.EfficientNetB7(
        input_shape=(512, 512, 3),
        weights='imagenet',
        include_top=False
    )
    # acc means val_acc here, not train acc.
    # Full training acc -> 95.4% (10epochs)
    # fine-tune at b7a acc -> 93.59% 12epochs
    # b6a loss -> 0.2687 - acc: 0.9467 11 epochs
    # b5a loss -> 0.2417 - acc: 0.9537 12e pochs
    # b4a loss -> 0.2411 - acc: 0.9539 12 epochs
    # b2a loss -> 0.2212 - acc: 0.9582 13 epochs
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block2a_expand_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # dropout : val_acc
    # 0 : 95.8
    # 0.2 : 95.82
    # Conclusion: No improved adding dropout at classifier.

    # 第四轮lr不能大于2e-4,后面几轮学习率必须每一轮都要下降
    model = tf.keras.Sequential([
        conv_base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2), #可有可无的dropout，但多少好那么一点点。。
        tf.keras.layers.Dense(104, activation='softmax')
    ])
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )


# 设置回调函数，当验证loss不再下降2个epochs时，stopping
# 设置学习率调度器，采用cos辅助下降
def lrfn(epoch):
    LR_START = 2.5e-4
    LR_MIN = 1e-5

    lr = cos(epoch*pi/18) * LR_START

    return lr if lr>LR_MIN else LR_MIN

def lrfn1(epoch):
    lr_start = 2.5e-4
    lr_min = 1e-5

    lr = lr_start / (1 + e ** (epoch - 5))
    return lr if lr > lr_min else lr_min

callbacks_list = [
    tf.keras.callbacks.LearningRateScheduler(
        lrfn1,
        verbose=1
    )
]


#开 始 训 练 ?
nums_image = count_data_items(TRAINING_FILENAMES)
STEPS_PER_EPOCH = nums_image // BATCH_SIZE
print('Total {} images will be training..'.format(nums_image))

history = model.fit(get_training_dataset(),  #给出训练数据(已经批处理了，且打乱了)
                    steps_per_epoch=STEPS_PER_EPOCH, #每个epoch所要重复获取数据然后训练的次数
                    epochs=EPOCHS,           #训练次数
                    validation_data=get_validation_dataset(),  #使用验证集检查过拟合！
                    callbacks=callbacks_list,
                    verbose=2
                    )

print('Model saving...')
model.save('enetb7_tf',save_format='tf')
#show_acc_loss(history)
# 加载验证集
print('Preparing dataset...')
dataset = get_validation_dataset(order=True)
images = dataset.map(lambda image, label: image)
labels = dataset.map(lambda image, label: label).unbatch()

# 用验证集预测
print('Predicting...')
prediction_start = perf_counter()
output = model.predict(images) # output.shape = (3712,104)
prediction_end = perf_counter()
print('In {:.1f}s compeleted!'.format(prediction_end - prediction_start))
prediction = np.argmax(output,axis=-1)

# 直接获得正确标签整个整体，batch的话还要写多一个循环
nums_validation_images = count_data_items(VALIDATION_FILENAMES)
correct_labels = next(iter(labels.batch(nums_validation_images))).numpy()

mistake = [0 for _ in range(104)] # 104 classes
print('Computing mistakes...')
for i in range(nums_validation_images):
    if prediction[i] != correct_labels[i]:
        mistake[correct_labels[i]]+=1

with open('mistake.json','w') as f:
    json.dump(mistake,f)

classes = [x for x in range(104)]
plt.bar(classes,mistake)
plt.xlabel('CLASSES')
plt.ylabel('Mistake count')
plt.title('Mistakes each class')
plt.savefig('mistake.jpeg')
plt.show()



#prediction test
c='''
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
'''

#confusion matrix
c1='''
cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
cm_probabilities = model.predict(images_ds)
cm_predictions = np.argmax(cm_probabilities, axis=-1)

cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
display_confusion_matrix(cmat, score, precision, recall)
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
'''
c2='''
def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
'''