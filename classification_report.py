from absl import flags, app
from absl.flags import FLAGS

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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


def main(_argv):
    # Load dataset
    test_set, class_names, test_length = data.dataset.load(split="test",
                                               size=FLAGS.size,
                                               batch_size=FLAGS.batch_size,
                                               cache=False)

    test_set = test_set.take(15)
    
    numpy_images = np.empty((0, FLAGS.size, FLAGS.size, 3))
    numpy_labels = np.empty((0, len(class_names)))
    for images, labels in test_set:
        numpy_images = np.vstack((numpy_images, images))
        numpy_labels = np.vstack((numpy_labels, labels))
    
    # print(numpy_images.shape, numpy_labels.shape)

    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size)

    # Compile the model
    metrics = ["accuracy"]
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=metrics)
    
    predictions = model.predict(numpy_images,
                                batch_size=FLAGS.batch_size)
    
    y_true = np.argmax(numpy_labels, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred),
                               class_names,
                               class_names)
    sns.heatmap(conf_matrix, xticklabels=True, yticklabels=True)
    plt.show()

if __name__ == "__main__":
    app.run(main)
