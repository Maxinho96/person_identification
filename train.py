from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras

import data.dataset


flags.DEFINE_boolean("fcn",
                     False,
                     "Fully Convolutional Network or standard network")


def main(_argv):
    d = data.dataset.get_datasets()[0]
    for f in d.take(5):
        print(f.numpy())


if __name__ == "__main__":
    app.run(main)
