from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras

import os

import data.dataset
import models


flags.DEFINE_integer("exp",
                     -1,
                     "The experiment number.")
flags.DEFINE_string("checkpoints_dir",
                    "checkpoints",
                    "The base dir where to save checkpoints.")
flags.DEFINE_string("logs_dir",
                    "logs",
                    "The base dir where to save TensorBoard logs.")
flags.DEFINE_integer("size",
                     None,
                     "Input image size. If None, the model accepts any size.")
flags.DEFINE_integer("trainable_layers",
                     1,
                     "Number of top layers to unfreeze and train.")
flags.DEFINE_string("optimizer",
                    "adam",
                    "The Keras optimizer to use.")
flags.DEFINE_integer('batch_size',
                     8,
                     'Batch size.')
flags.DEFINE_integer("epochs",
                     1,
                     "Number of epochs.")
flags.DEFINE_integer("patience",
                     10,
                     "Early stopping patience.")
flags.DEFINE_integer("initial_epoch",
                     0,
                     "The epoch number to start from.")
flags.DEFINE_boolean("cache_train",
                     True,
                     "Cache the training set in RAM or not.")
flags.DEFINE_boolean("cache_val",
                     True,
                     "Cache the validation set in RAM or not.")
flags.DEFINE_string("weights",
                    "imagenet",
                    "one of `None` (random initialization), \
                    'imagenet' (pre-training on ImageNet), \
                    or the path to the weights file to be loaded.")


def main(_argv):
    # Load datasets
    train_set, class_names, train_length = data.dataset.load(
        split="train",
        size=FLAGS.size,
        batch_size=FLAGS.batch_size,
        cache=FLAGS.cache_train
    )
    val_set, _, val_length = data.dataset.load(split="val",
                                               size=FLAGS.size,
                                               batch_size=FLAGS.batch_size,
                                               cache=FLAGS.cache_val)

    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size,
                             weights=FLAGS.weight_weights)

    # Freeze bottom layers
    for layer in model.layers[:-FLAGS.trainable_layers]:
        if not "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = False
    print("Trainable layers:")
    for layer in model.layers:
        if layer.trainable:
            print(layer.name)

    # Compile the model
    metrics = ["accuracy",
               keras.losses.CategoricalCrossentropy(
                   name="categorical_crossentropy")
               ]
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=metrics)

    # Define callbacks
    callbacks = []

    # Checkpoint callback
    filepath = os.path.join(FLAGS.checkpoints_dir,
                            "exp" + str(FLAGS.exp),
                            "best_weights.ckpt")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True)
    callbacks.append(checkpoint_cb)

    # Early stopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                      patience=FLAGS.patience)
    callbacks.append(early_stopping_cb)

    # TensorBoard callback
    log_dir = os.path.join(FLAGS.logs_dir, "exp" + str(FLAGS.exp))
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir)
    callbacks.append(tensorboard_cb)

    # Fit the model
    model.fit(train_set,
              epochs=FLAGS.epochs,
              callbacks=callbacks,
              validation_data=val_set,
              initial_epoch=FLAGS.initial_epoch,
              steps_per_epoch=train_length // 20 // FLAGS.batch_size,
              validation_steps=val_length // 50 // FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)
