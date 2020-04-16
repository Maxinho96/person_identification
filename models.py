import tensorflow.keras as keras


def get_standard_model(num_classes):
    base_model = keras.applications.xception.Xception(
        # keras.applications.nasnet.NASNetLarge(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    inputs = base_model.input
    base_outputs = base_model.output
    outputs = keras.layers.Dense(units=num_classes,
                                 activation="softmax",
                                 name="output")(base_outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def get_fcn_model(num_classes):
    base_model = keras.applications.xception.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(None, None, 3)
    )

    inputs = base_model.input
    base_outputs = base_model.output
    conv = keras.layers.Conv2D(filters=num_classes,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               name="conv1x1")(base_outputs)
    outputs = keras.layers.GlobalAveragePooling2D(name="avg")(conv)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
