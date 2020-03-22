import tensorflow.keras as keras


def get_standard_model(num_classes):
    base_model = keras.applications.nasnet.NASNetLarge(weights="imagenet",
                                                       include_top=False)

    inputs = base_model.input
    base_outputs = base_model.output
    avg = keras.layers.GlobalAveragePooling2D(name="avg")(base_outputs)
    outputs = keras.layers.Dense(units=num_classes,
                                 activation="softmax",
                                 name="output")(avg)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def get_fcn_model(num_classes):
    inputs = keras.Input(shape=(None, None, 3), name="input")

    base_model = keras.applications.nasnet.NASNetLarge(weights="imagenet",
                                                       include_top=False)
    base_model.layers.pop(0)
    base_outputs = base_model(inputs).output

    # inputs = base_model.input
    # base_outputs = base_model_newinput.output
    conv = keras.layers.Conv2D(filters=num_classes,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               name="conv1x1")(base_outputs)

    avg = keras.layers.GlobalAveragePooling2D(name="avg")(conv)
    outputs = avg

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
