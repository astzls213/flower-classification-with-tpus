#导入核心库
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
import numpy as np
import matplotlib.pyplot as plt
import re

try:
    #解析TPU芯片集群
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    #kaggle不需要喂参数，他默认好了
    tf.config.experimental_connect_to_cluster(tpu)
    # 配置运算器
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # 初始化tpu系统
except ValueError: #本地运行，CPU
    tpu = None

#获得分配策略，进行优化，以提升训练速度
if tpu:
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

#后面很多核心函数都有优化参数，根据性能自动选择
AUTO = tf.data.experimental.AUTOTUNE
#超参数定义
#strategy来自tpu_settings.py
BATCH_SIZE = 16 * strategy.num_replicas_in_sync # 根据cpu/tpu自动调整batch大小
LEARNING_RATE = 1e-4
EPOCHS = 8 # 训练周次


IMAGE_SIZE = [512,512] #手动修改此处图像大小，进行训练
WIDTH = IMAGE_SIZE[0]
HEIGHT = IMAGE_SIZE[1]
CHANNELS = 3
CLASS_WEIGHT = {
    0: 1.0, 1: 2.168, 2: 2.26, 3: 2.243, 4: 1.0, 5: 1.746, 6: 2.297, 7: 1.68, 8: 1.746, 9: 1.758, 
    10: 1.0, 11: 1.992, 12: 1.726, 13: 1.0, 14: 1.0, 15: 2.243, 16: 1.906, 17: 1.94, 18: 1.734, 19: 2.168, 
    20: 2.278, 21: 1.711, 22: 1.954, 23: 2.278, 24: 1.754, 25: 1.762, 26: 2.243, 27: 2.075, 28: 1.636, 29: 1.667, 
    30: 1.68, 31: 2.196, 32: 2.211, 33: 2.26, 34: 2.297, 35: 2.055, 36: 1.894, 37: 2.168, 38: 2.278, 39: 1.807, 
    40: 1.853, 41: 1.711, 42: 1.859, 43: 1.664, 44: 2.297, 45: 1.0, 46: 1.0, 47: 1.0, 48: 1.0, 49: 1.0, 
    50: 1.0, 51: 1.68, 52: 1.648, 53: 1.0, 54: 2.045, 55: 1.888, 56: 1.738, 57: 1.859, 58: 2.055, 59: 1.888, 
    60: 2.155, 61: 2.13, 62: 1.722, 63: 2.142, 64: 1.906, 65: 2.085, 66: 2.243, 67: 1.0, 68: 1.0, 69: 1.719, 
    70: 1.683, 71: 1.0, 72: 1.0, 73: 1.0, 74: 1.0, 75: 1.0, 76: 1.636, 77: 1.0, 78: 1.75, 79: 1.639, 
    80: 1.0, 81: 1.693, 82: 1.0, 83: 1.657, 84: 2.107, 85: 2.13, 86: 1.0, 87: 1.0, 88: 1.711, 89: 1.969, 
    90: 1.677, 91: 1.66, 92: 2.196, 93: 1.0, 94: 1.0, 95: 1.0, 96: 1.697, 97: 2.009, 98: 2.075, 99: 2.196, 
    100: 2.107, 101: 2.182, 102: 1.0, 103: 1.0
}
CLASS_WEIGHT_TPU_VER1 = [
    0, 1.0, 1, 2.168, 2, 2.26, 3, 2.243, 4, 1.0, 5, 1.746, 6, 2.297, 7, 1.68, 8, 1.746, 9, 1.758, 
    10, 1.0, 11, 1.992, 12, 1.726, 13, 1.0, 14, 1.0, 15, 2.243, 16, 1.906, 17, 1.94, 18, 1.734, 19, 2.168, 
    20, 2.278, 21, 1.711, 22, 1.954, 23, 2.278, 24, 1.754, 25, 1.762, 26, 2.243, 27, 2.075, 28, 1.636, 29, 1.667, 
    30, 1.68, 31, 2.196, 32, 2.211, 33, 2.26, 34, 2.297, 35, 2.055, 36, 1.894, 37, 2.168, 38, 2.278, 39, 1.807, 
    40, 1.853, 41, 1.711, 42, 1.859, 43, 1.664, 44, 2.297, 45, 1.0, 46, 1.0, 47, 1.0, 48, 1.0, 49, 1.0, 
    0, 1.0, 51, 1.68, 52, 1.648, 53, 1.0, 54, 2.045, 55, 1.888, 56, 1.738, 57, 1.859, 58, 2.055, 59, 1.888, 
    60, 2.155, 61, 2.13, 62, 1.722, 63, 2.142, 64, 1.906, 65, 2.085, 66, 2.243, 67, 1.0, 68, 1.0, 69, 1.719, 
    70, 1.683, 71, 1.0, 72, 1.0, 73, 1.0, 74, 1.0, 75, 1.0, 76, 1.636, 77, 1.0, 78, 1.75, 79, 1.639, 
    80, 1.0, 81, 1.693, 82, 1.0, 83, 1.657, 84, 2.107, 85, 2.13, 86, 1.0, 87, 1.0, 88, 1.711, 89, 1.969, 
    90, 1.677, 91, 1.66, 92, 2.196, 93, 1.0, 94, 1.0, 95, 1.0, 96, 1.697, 97, 2.009, 98, 2.075, 99, 2.196, 
    100, 2.107, 101, 2.182, 102, 1.0, 103, 1.0
] # Ver1 using factor * log(frequency)
CLASS_WEIGHT_TPU_VER2 = [0, 2.879, 1, 30.117, 2, 39.152, 3, 37.287, 4, 1.114, 5, 9.0, 6, 43.502, 7, 7.457, 8, 9.0, 9, 9.322, 
    10, 5.758, 11, 18.21, 12, 8.511, 13, 2.977, 14, 3.449, 15, 37.287, 16, 14.237, 17, 15.661, 18, 8.7, 19, 30.117, 
    20, 41.212, 21, 8.157, 22, 16.313, 23, 41.212, 24, 9.212, 25, 9.434, 26, 37.287, 27, 23.03, 28, 6.58, 29, 7.184, 
    30, 7.457, 31, 32.626, 32, 34.045, 33, 39.152, 34, 43.502, 35, 21.751, 36, 13.737, 37, 30.117, 38, 41.212, 39, 10.726, 
    40, 12.235, 41, 8.157, 42, 12.429, 43, 7.118, 44, 43.502, 45, 4.553, 46, 6.264, 47, 3.0, 48, 1.856, 49, 1.391, 
    50, 3.896, 51, 7.457, 52, 6.809, 53, 1.702, 54, 21.163, 55, 13.501, 56, 8.798, 57, 12.429, 58, 21.751, 59, 13.501, 
    60, 29.001, 61, 27.001, 62, 8.42, 63, 27.966, 64, 14.237, 65, 23.728, 66, 37.287, 67, 1.001, 68, 3.012, 69, 8.33, 
    70, 7.529, 71, 5.716, 72, 4.689, 73, 1.702, 74, 6.264, 75, 2.559, 76, 6.58, 77, 5.633, 78, 9.105, 79, 6.636, 
    80, 5.118, 81, 7.753, 82, 5.844, 83, 6.991, 84, 25.259, 85, 27.001, 86, 6.525, 87, 5.363, 88, 8.157, 89, 17.022, 
    90, 7.387, 91, 7.054, 92, 32.626, 93, 5.633, 94, 5.977, 95, 6.166, 96, 7.83, 97, 19.098, 98, 23.03, 99, 32.626, 
    100, 25.259, 101, 31.321, 102, 2.008, 103, 1.054
]# Ver2 using factor * frequency

#文件路径构造
try: 
    # Running on Kaggle kernel
    from kaggle_datasets import KaggleDatasets
    OFFICIAL_BASE = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')
    EXTERNAL_BASE = KaggleDatasets().get_gcs_path('oxford-flowers-tfrecords')
except ModuleNotFoundError: 
    # Running on my mac
    OFFICIAL_BASE = "/Users/astzls/Downloads/flower"
    EXTERNAL_BASE = None # 没下载= =

OFFICIAL_PATH = { # 根据图像大小选择路径
    192: OFFICIAL_BASE + '/tfrecords-jpeg-192x192',
    224: OFFICIAL_BASE + '/tfrecords-jpeg-224x224',
    331: OFFICIAL_BASE + '/tfrecords-jpeg-331x331',
    512: OFFICIAL_BASE + '/tfrecords-jpeg-512x512'
}
EXTERNAL_PATH = {
    192: EXTERNAL_BASE + '/tfrecords-png-192x192',
    224: EXTERNAL_BASE + '/tfrecords-png-224x224',
    331: EXTERNAL_BASE + '/tfrecords-png-331x331',
    512: EXTERNAL_BASE + '/tfrecords-png-512x512'
}
OFFICIAL_IMAGE_PATH = OFFICIAL_PATH[IMAGE_SIZE[0]]

#此处利用tf.io的库函数
#读出文件集方式很多种，也可以用os+re库进行，但没有这个方便

OFFICIAL_TRAINING_FILENAMES = tf.io.gfile.glob(OFFICIAL_IMAGE_PATH + '/train/*.tfrec')
OFFICIAL_VALIDATION_FILENAMES = tf.io.gfile.glob(OFFICIAL_IMAGE_PATH + '/val/*.tfrec')
OFFICIAL_TEST_FILENAMES = tf.io.gfile.glob(OFFICIAL_IMAGE_PATH + '/test/*.tfrec')

EXTERNAL_FILENAMES = tf.io.gfile.glob(EXTERNAL_PATH[IMAGE_SIZE[0]] + '/*.tfrec')
# 统计图片数量
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

#下面是核心函数构造，自顶向下
def get_training_dataset(external=False):
    # external control whether loading external dataset or not.
    # external 用来决定是否加载来自牛津大学的花蕊数据，共7300多张。
    if external:
        external_dataset = load_dataset(EXTERNAL_FILENAMES, labeled=True, external=True)

    # 加载官方给定数据
    dataset = load_dataset(OFFICIAL_TRAINING_FILENAMES, labeled=True, external=False)
    #labeled参数是用来判断数据有没有label的，比如test dataset就没有label
    #装载分离好等数据，每个example含image，class

    #连接两个数据集
    if external:
        dataset = dataset.concatenate(external_dataset)

    dataset = dataset.map(data_augmentation, num_parallel_calls=AUTO)
    #进行数据扩容，此步非常重要！可大大提升训练精度，已被广泛使用，不用你就out了

    dataset = dataset.shuffle(2048)
    #打乱，2048是根据data的数量决定的。可以先写个函数跑跑究竟有多少图片可供训练

    dataset = dataset.batch(BATCH_SIZE)
    #批处理

    dataset = dataset.repeat()
    #repeat无参数说明无限重复dataset。放心，不会内存溢出，只是懒循环罢了

    dataset = dataset.prefetch(AUTO)
    #根据cpu决定是否提前准备数据。为什么要这么做？原因是我想采用tpu进行训练，
    # 那么在tpu在训练时，cpu可以预先把下一批图像load到内存，当
    # tpu训练好了，直接又能进行下一批当训练，减少了训练时间。
    # 还是那句话，不用，你就out了
    return dataset

def get_validation_dataset():
    dataset = load_dataset(OFFICIAL_VALIDATION_FILENAMES, labeled=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache() #cache dataset到memory，加速训练时间，只有3712张
    dataset = dataset.prefetch(AUTO) 
    return dataset

def get_test_dataset():
    dataset = load_dataset(OFFICIAL_TEST_FILENAMES, labeled=False)
    dataset = dataset.batch(BATCH_SIZE)
    #dataset = dataset.cache() #可cache也可以步cache，随便。但是有7382个测试图片，小心内存溢出
    dataset = dataset.prefetch(AUTO) 
    return dataset

#完善核心函数中的小函数
def load_dataset(filenames, labeled=True, ordered=False, external=False):
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
    if external:
        dataset = dataset.map(read_png_tfrecord,num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_labeled_tfrecord if labeled
                            else read_unlabeled_tfrecord, 
                            num_parallel_calls=AUTO)

    #现在dataset的每个data有两个部分了，一个是image，一个是class或id号
    return dataset 

def data_augmentation(image, label):
    #所谓数据扩容，就是把原来的照片左移或右移，或上下左右反转一下，就得到了“新”图像
    #此方法利用了现实世界的平移不变形和空间不变形

    # k取值的几种情况
    # 0 不转 1 逆时针90 2 逆时针180 3 逆时针270
    k = np.random.randint(4)
    image = tf.image.rot90(image,k=k)
    # 饱和度随机
    #image = tf.image.random_saturation(image, lower=0, upper=3)

    # 还有下列api进行更无聊的处理
    #'random_brightness',  亮度
    # 'random_contrast', 对比度
    # 'random_crop', 缩放
    # 'random_flip_up_down', 上下翻转
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

def read_png_tfrecord(example):
    FEATURE = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64)  
    }
    example = tf.io.parse_single_example(example, FEATURE)

    image = tf.image.decode_png(example['image'], channels=CHANNELS)
    #得到tf.Tensor形式的image,内部类型为uint8 or uint16

    image = tf.cast(image, tf.float32)
    image /= 255.0
    #将image的每个数值调整在[0,1]之间，方便训练

    image = tf.reshape(image, [*IMAGE_SIZE, CHANNELS])
    #reshape此部非常重要，调试的时候被坑了，老是说什么shape不匹配

    label = tf.cast(example['class'], tf.int32)
    return image, label


def show_acc_loss(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy: GAP | lr: {} | Adam '.format(LEARNING_RATE))
    plt.legend()
    plt.savefig('acc.jpg')

    plt.figure()

    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss & CE')
    plt.title('Loss: GAP | lr: {} | Adam '.format(LEARNING_RATE))
    plt.legend()

    plt.savefig('loss.jpg')

    plt.show()
    
#核心函数定义
def res():
    with strategy.scope():
        conv_base = ResNet50V2(weights='imagenet',
                                include_top=False ,
                                input_shape=[*IMAGE_SIZE, 3])
        conv_base.trainable = True
        
        set_trainable = False
        for layer in conv_base.layers:
            #c5b2_acc:
            if layer.name == 'conv5_block2_preact_bn':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        input = tf.keras.layers.Input(shape=(512,512,3))  
        res = conv_base(input)
        gap = tf.keras.layers.GlobalAveragePooling2D()(res)
        #dropout = tf.keras.layers.Dropout(0.2)(gap)
        output = tf.keras.layers.Dense(104, activation='softmax')(gap)

        model = tf.keras.models.Model(input,output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

# 创建网络       
model = res()

# 设置回调函数，当验证loss不再下降2个epochs时，stopping
# 当验证损失下降当很慢当时候，下调lr
def lr_scheduler(epoch):
    LR_START = 9e-5
    LR_MAX = 1e-4
    LR_MIN = 4.8e-5
    LR_RAMPUP_EPOCHS = 3
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = 0.6
    
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    else:
        lr =  3.5e-5 * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2
    )
]


#开 始 训 练 ?
if tpu:
    use_external_dataset = True
    nums_file = count_data_items(OFFICIAL_TRAINING_FILENAMES)
    if use_external_dataset:
        nums_file += count_data_items(EXTERNAL_FILENAMES)
    print('Total {} data'.format(nums_file))
    history = model.fit(get_training_dataset(external=use_external_dataset),  #给出训练数据(已经批处理了，且打乱了)
                        steps_per_epoch=nums_file // BATCH_SIZE, #每个epoch所要重复获取数据然后训练的次数
                        epochs=EPOCHS,           #训练次数
                        validation_data=get_validation_dataset(),  #使用验证集检查过拟合！
                        callbacks=callbacks_list,
                        #class_weight=CLASS_WEIGHT_TPU_VER2,
                        verbose=2
                        )
    model.save('best.h5')

    show_acc_loss(history)

