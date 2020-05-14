from absl import flags, app
from absl.flags import FLAGS

from sklearn.metrics import classification_report
import numpy as np

import data.dataset
import models


flags.DEFINE_integer("size",
                     None,
                     "Input image size. If None, the model accepts any size.")
flags.DEFINE_string("optimizer",
                    "adam",
                    "The Keras optimizer to use.")
flags.DEFINE_integer('batch_size',
                     8,
                     'Batch size.')
flags.DEFINE_boolean("cache_test",
                     True,
                     "Cache the test set in RAM or not")


def main(_argv):
    # Load dataset
    test_set, class_names, test_length = data.dataset.load(split="test",
                                               size=FLAGS.size,
                                               batch_size=FLAGS.batch_size,
                                               cache=FLAGS.cache_test)

    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size)

    # Compile the model
    metrics = ["accuracy"]
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=metrics)

    # Evaluate the model
    model.evaluate(test_set,
                   steps=test_length // FLAGS.batch_size // 1000)


if __name__ == "__main__":
    app.run(main)
