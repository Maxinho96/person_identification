from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras
import tensorflow as tf

import os

import data.dataset
import models


flags.DEFINE_integer("size",
                     None,
                     "Input image size. If None, the model accepts any size.")
flags.DEFINE_string("optimizer",
                    "adam",
                    "The Keras optimizer to use.")
flags.DEFINE_boolean("cache_test",
                     True,
                     "Cache the test set in RAM or not")


def main(_argv):
    # Load dataset
    test_set, class_names, test_length = data.dataset.load(split="test",
                                               size=FLAGS.size,
                                               batch_size=1,
                                               cache=FLAGS.cache_test)

    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size)

    # Compile the model
    metrics = ["accuracy",
               keras.losses.CategoricalCrossentropy(
                   name="categorical_crossentropy")
               ]
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=metrics)


    # Evaluate the model
    model.evaluate(test_set)


if __name__ == "__main__":
    app.run(main)
