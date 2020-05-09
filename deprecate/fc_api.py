import tensorflow as tf
def VGG_via_FC():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[512,512, 3])
    conv_base.trainable = True # False = transfer learning, True = fine-tuning
        
    #冻结前4个block，解冻后面对block
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block4_conv2':
            set_trainable = True
        if set_trainable:
           layer.trainable = True
        else:
            layer.trainable = False
        
        #conv_base.summary()
    x = tf.keras.layers.Input(shape=(512,512,3))  
    dp1 = tf.keras.layers.Dropout(0.5)(x)
    vgg = conv_base(dp1)
    dp2 = tf.keras.layers.Dropout(0.5)(vgg)
    gap = tf.keras.layers.GlobalAveragePooling2D()(dp2)
    dp3 = tf.keras.layers.Dropout(0.3)(gap)
    output = tf.keras.layers.Dense(104, activation='softmax')(dp3)

    model = tf.keras.models.Model(x,output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

z=VGG_via_FC()
z.summary()