    ''' 过拟合模型
    rtv = tf.keras.models.Sequential() # 使用最传统的线性网络
    #接下来添加可分离的2D卷积层，这东西keras作者说比Conv2D好用，不知道真假。。
    rtv.add(tf.keras.layers.Conv2D(64,3,data_format="channels_last",activation="relu",padding="same",input_shape=(WIDTH,HEIGHT,CHANNELS)))
    rtv.add(tf.keras.layers.Conv2D(64,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.Conv2D(128,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(128,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.Conv2D(256,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(256,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))    
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.Conv2D(512,3,data_format="channels_last",activation="relu",padding="same"))
    rtv.add(tf.keras.layers.MaxPooling2D(2))

    rtv.add(tf.keras.layers.GlobalAveragePooling2D())
    rtv.add(tf.keras.layers.Dense(512,activation="relu"))
    rtv.add(tf.keras.layers.BatchNormalization())
    rtv.add(tf.keras.layers.Dense(104,activation="softmax"))
    ‘’‘