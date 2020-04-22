from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras
import matplotlib.pyplot as plt

import os

import data.dataset
import models


flags.DEFINE_integer("exp",
                     -1,
                     "The experiment number.")
flags.DEFINE_string("checkpoints_dir",
                    "checkpoints",
                    "The base dir where to save checkpoints.")
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
flags.DEFINE_boolean("cache_train",
                     True,
                     "Cache the training set in RAM or not.")
flags.DEFINE_boolean("cache_val",
                     True,
                     "Cache the validation set in RAM or not.")


def main(_argv):
    # Load datasets
    train_set, class_names = data.dataset.load(split="train",
                                               size=FLAGS.size,
                                               batch_size=8,
                                               cache=FLAGS.cache_train)
    val_set, _ = data.dataset.load(split="val",
                                   size=FLAGS.size,
                                   batch_size=8,
                                   cache=FLAGS.cache_val)

    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size)

    # Freeze bottom layers
    for layer in model.layers[:-FLAGS.trainable_layers]:
        layer.trainable = False
    print("Trainable layers:")
    for layer in model.layers[-FLAGS.trainable_layers:]:
        print(layer.name)

    # Compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=["accuracy"])

    # Define callbacks
    callbacks = []
    # Checkpoint callback
    filepath = os.path.join(FLAGS.checkpoints_dir,
                            "exp"+str(FLAGS.exp),
                            "best_weights.ckpt")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        save_weights_only=True)
    callbacks.append(checkpoint_cb)
    # Early stopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
    callbacks.append(early_stopping_cb)
    # TensorBoard callback
    
    # for image_batch, label_batch in train_set:
    #     data.dataset.show_batch(image_batch.numpy(),
    #                             label_batch.numpy(),
    #                             class_names)
    #     if input() == "q":
    #         break


if __name__ == "__main__":
    app.run(main)
