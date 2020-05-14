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

    # test_set = test_set.take(test_length // 50 // FLAGS.batch_size)
    
    # Load model
    model = models.get_model(num_classes=len(class_names),
                             size=FLAGS.size)

    # Compile the model
    metrics = ["accuracy"]
    model.compile(loss="categorical_crossentropy",
                  optimizer=FLAGS.optimizer,
                  metrics=metrics)
    
    y_true = np.empty((0,))
    y_pred = np.empty((0,))
    for images, labels in test_set:
        predictions = model.predict(images)
        curr_y_true = np.argmax(labels, axis=1)
        y_true = np.concatenate((y_true, curr_y_true))
        curr_y_pred = np.argmax(predictions, axis=1)
        y_pred = np.concatenate((y_pred, curr_y_pred))
    
    # print(numpy_images.shape, numpy_labels.shape)
    
    print(classification_report(y_true, y_pred, target_names=class_names))
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred),
                               class_names,
                               class_names)
    sns.heatmap(conf_matrix, xticklabels=True, yticklabels=True, cmap="YlGnBu")
    plt.show()

if __name__ == "__main__":
    app.run(main)
