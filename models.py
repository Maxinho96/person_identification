from absl import app

import tensorflow.keras as keras


# Get an Xception model. It can be of a fixed size or accept any size.
# The model is the same, but fixed size is more efficient if you don't
# plan to use the model on different sizes.
def get_model(num_classes, size=None, weights="imagenet"):
    if weights != "imagenet":
        initial_weights = None
        final_weights = weights
    else:
        initial_weights = weights
        final_weights = None

    base_model = keras.applications.xception.Xception(
        # keras.applications.nasnet.NASNetLarge(
        weights=initial_weights,
        include_top=False,
        input_shape=(size, size, 3),
        pooling="avg"
    )

    inputs = base_model.input
    base_outputs = base_model.output
    outputs = keras.layers.Dense(units=num_classes,
                                 activation="softmax",
                                 name="output")(base_outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    if final_weights is not None:
        model.load_weights(final_weights)

    return model


def main(_argv):
    m = get_model(14)
    print(m.summary())


if __name__ == "__main__":
    app.run(main)



# def get_fcn_model(num_classes):
#     base_model = keras.applications.xception.Xception(
#         weights="imagenet",
#         include_top=False,
#         input_shape=(None, None, 3)
#     )

#     inputs = base_model.input
#     base_outputs = base_model.output
#     conv = keras.layers.Conv2D(filters=num_classes,
#                                kernel_size=1,
#                                strides=1,
#                                padding="same",
#                                name="conv1x1")(base_outputs)
#     avg = keras.layers.GlobalAveragePooling2D(name="avg")(conv)
#     outputs = keras.activations.softmax(avg)

#     model = keras.Model(inputs=inputs, outputs=outputs)

#     return model
