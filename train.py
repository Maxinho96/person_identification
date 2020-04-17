from absl import flags, app
from absl.flags import FLAGS

import tensorflow.keras as keras
import matplotlib.pyplot as plt

import data.dataset


flags.DEFINE_boolean("fcn",
                     False,
                     "Fully Convolutional Network or standard network")
flags.DEFINE_integer('batch_size', 8, 'Batch size')


def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10, 10))
    batch_size = image_batch.shape[0]
    for n in range(batch_size):
        _ = plt.subplot(1, batch_size, n + 1)
        image = image_batch[n]
        plt.imshow(image / 2 + 0.5)
        plt.title(class_names[label_batch[n]][0]+"\n"+str(image.shape))
        plt.axis('off')
    plt.show()


def main(_argv):
    dataset, class_names = data.dataset.load()
    dataset = iter(dataset)
    while input() != "q":
        image_batch, label_batch = next(dataset)
        show_batch(image_batch.numpy(), label_batch.numpy(), class_names)

if __name__ == "__main__":
    app.run(main)
