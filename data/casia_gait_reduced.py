from absl import app, flags
from absl.flags import FLAGS
import os
import numpy as np
from shutil import copyfile
from functools import partial
import pdb
from sklearn.model_selection import train_test_split

flags.DEFINE_string("dataset_dir", "casia_gait/DatasetB_split",
                    "Path to full CASIA Gait split dataset")
flags.DEFINE_string("output_dir", "casia_gait/DatasetB_split_reduced",
                    "Target directory to save the reduced dataset")


def copy(path, split):
    *_, folder, basename = path.split(os.path.sep)

    dest_dir = os.path.join(FLAGS.output_dir, split, folder)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    copyfile(src=path,
             dst=os.path.join(dest_dir, basename))


def main(_argv):

    all_paths = np.empty([0, 1])
    all_labels = np.empty([0, 1])
    for root_dir, _, files in os.walk(FLAGS.dataset_dir):
        # i=0
        for file_name in files:
            if file_name.endswith(".jpg"):
                path = os.path.join(root_dir, file_name)

                *_, folder, basename = path.split(os.path.sep)
                _, _, seq_num, *_ = basename.split("-")

                if seq_num == "01":
                    all_paths = np.append(all_paths, path)
                    all_labels = np.append(all_labels, folder)

                    print("Loading", file_name)
                    # i+=1
                    # if i == 4:
                    #     break

    np.random.shuffle(all_paths)

    train_val_paths, test_paths, train_val_labels, test_labels = \
        train_test_split(all_paths,
                         all_labels,
                         test_size=1 / 10,
                         stratify=all_labels)
    train_paths, val_paths = train_test_split(train_val_paths,
                                              test_size=1 / 9,
                                              stratify=train_val_labels)

    train_fn = np.vectorize(partial(copy, split="train"))
    val_fn = np.vectorize(partial(copy, split="val"))
    test_fn = np.vectorize(partial(copy, split="test"))
    train_fn(train_paths)
    val_fn(val_paths)
    test_fn(test_paths)


if __name__ == "__main__":
    app.run(main)
