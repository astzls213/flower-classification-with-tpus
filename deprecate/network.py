#import 核心库
import tensorflow as tf

from tpu_settings import *
from loading import *
'''
这里是一些层的定义，为了方便使用

'''
#超参数定义
#strategy来自tpu_settings.py
BATCH_SIZE = 16 * strategy.num_replicas_in_sync # 根据cpu/tpu自动调整batch大小
LEARNING_RATE = 1e-3
EPOCHS = 12 # 训练周次
STEPS_PER_EPOCH = 12753 // BATCH_SIZE
IMAGE_SIZE = [512,512] #手动修改此处图像大小，进行训练
WIDTH = IMAGE_SIZE[0]
HEIGHT = IMAGE_SIZE[1]
CHANNELS = 3
#导入实用函数及变量

from loading import *

#包装函数定义
def get_Conv2D(filters,size):
    #size 是卷积核的边长
    #默认步幅
    return tf.keras.layers.Conv2D(filters,size,data_format="channels_last",activation="relu")

def get_MaxPooling(size):
    #size 是池化器的边长
    return tf.keras.layers.MaxPooling2D(size)

#核心函数定义
def ZNN(): # ZNN 就是本人设计的网络，哈哈
    rtv = tf.keras.models.Sequential() # 使用最传统的线性网络
    #接下来添加可分离的2D卷积层，这东西keras作者说比Conv2D好用，不知道真假。。

    rtv.add(tf.keras.layers.Conv2D(64,3,data_format="channels_last",activation="relu",padding="same",input_shape=(WIDTH,HEIGHT,CHANNELS)))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(get_Conv2D(128,3)) #filters, size
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(get_Conv2D(256,3))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(get_Conv2D(512,3))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(get_Conv2D(512,3))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.GlobalAveragePooling2D())

    rtv.add(tf.keras.layers.Dense(104,activation="softmax"))

    #set optimizer^ ^ 使用rmsprop
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9)
    #compile section*_* 代价函数使用多分类交叉墒
    rtv.compile(optimizer=rmsprop,
                loss="sparse_categorical_crossentropy", #千万别用categorical_crossentropy，那个是为one-hot形式lable准备的。
                metrics=["sparse_categorical_accuracy"])
    return rtv

def vgg_via_fc():
    model = tf.keras.models.Sequential()
    conv_base = tf.keras.layers.Dense(100,activation='relu',input_shape=[*IMAGE_SIZE,CHANNELS])
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(104, activation='softmax'))
    
    #set optimizer^ ^ 使用rmsprop
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=LookupError, rho=0.9)
    #compile section*_* 代价函数使用多分类交叉墒
    model.compile(optimizer=rmsprop,
                loss="sparse_categorical_crossentropy", #千万别用categorical_crossentropy，那个是为one-hot形式lable准备的。
                metrics=["sparse_categorical_accuracy"])
    return model

if __name__ == "__main__":
    #创建网络
    with strategy.scope():
        model = vgg_via_fc()
    
    #检查一下ZZNN的参数是否过多，用于调试
    #model.summary()
    
    #开 始 训 练 ？
    history = model.fit(get_training_dataset(),  #给出训练数据(已经批处理了，且打乱了)
                        steps_per_epoch=STEPS_PER_EPOCH, #每个epoch所要重复获取数据然后训练的次数
                        epochs=EPOCHS,           #训练次数
                        validation_data=get_validation_dataset(),  #使用验证集检查过拟合！
                        )
