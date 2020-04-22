from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras
import matplotlib.pyplot as plt

import data.dataset


flags.DEFINE_integer("size",
                     None,
                     "Input image size. Can be None (any size).")
flags.DEFINE_integer('batch_size',
                     8,
                     'Batch size.')
flags.DEFINE_boolean("cache_train",
                     True,
                     "Cache the training set in RAM or not.")
flags.DEFINE_boolean("cache_val",
                     True,
                     "Cache the validation set in RAM or not.")


def main(_argv):
    training_set, class_names = data.dataset.load(split="train",
                                                  size=FLAGS.size,
                                                  batch_size=8,
                                                  cache=FLAGS.cache_train)

    for image_batch, label_batch in training_set:
        data.dataset.show_batch(image_batch.numpy(),
                                label_batch.numpy(),
                                class_names)
        if input() == "q":
            break


if __name__ == "__main__":
    app.run(main)
