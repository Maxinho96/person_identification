from absl import flags, app
from absl.flags import FLAGS

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import os
from functools import partial


flags.DEFINE_string("dataset",
                    "data/casia_gait/DatasetB_split",
                    "Path to dataset")
flags.DEFINE_boolean("random_flip",
                     True,
                     "Use or not random flip data augmentation")
flags.DEFINE_boolean("random_hue",
                     True,
                     "Use or not random hue data augmentation")
flags.DEFINE_boolean("random_saturation",
                     True,
                     "Use or not random saturation data augmentation")
flags.DEFINE_boolean("cache",
                     True,
                     "Cache the dataset in RAM or not")
# TODO: aggiungere tutti i flag di data augmentation


def preprocess(image, size):
    # Resize the image to the desired size.
    if size is not None:
        image = tf.image.resize_with_pad(image, size, size)
    # Convert uint8 image to float32 image
    image = tf.cast(image, tf.float32)
    # Preprocess input for Xception
    # (this wants an RGB float image in range [0., 255.], and gives
    # an RGB float image in range [-1., 1.]
    image = keras.applications.xception.preprocess_input(image)

    return image


def augment_and_preprocess(image, label, size):
    # Randomly flip the image horizontally.
    if FLAGS.random_flip:
        image = tf.image.random_flip_left_right(image)
    # Randomly adjust the image hue.
    if FLAGS.random_hue:
        image = tf.image.random_hue(image, 0.08)
        # Randomly adjust the image saturation.
    if FLAGS.random_saturation:
        image = tf.image.random_saturation(image, 0, 0.7)

    image = preprocess(image, size)

    return image, label


def decode_image(file_path, class_names, size, do_preprocessing):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label = parts[-2] == class_names
    # Load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Preprocessing can be done here or not. It must be done after data
    # augmentation for the train split.
    if do_preprocessing:
        image = preprocess(image, size)

    return image, label


# Returns the dataset split indicated in the split parameter.
# You can specify the interval of classes to use, setting first_class and
# last_class. You can also skip some classes of the interval setting
# skip_classes.
def load(split="train",
         first_class="001",
         last_class="014",
         skip_classes=(),
         size=None,
         batch_size=8):
    if split in os.listdir(FLAGS.dataset):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset_dir = os.path.join(FLAGS.dataset, split)

        class_names = sorted(os.listdir(dataset_dir))

        start = class_names.index(first_class)
        end = class_names.index(last_class) + 1
        class_names = class_names[start:end]
        class_names = [c for c in class_names if c not in skip_classes]

        # Load files.
        pattern = [os.path.join(dataset_dir, c, "*") for c in class_names]
        filenames_ds = tf.data.Dataset.list_files(pattern)

        # Preprocess images after decoding only if split is not train.
        # Preprocessing is done after augmentation if split is train.
        do_preprocessing = split != "train"
        # Decode and optionally preprocess images and add labels.
        decode_fn = partial(decode_image,
                            class_names=class_names,
                            size=size,
                            do_preprocessing=do_preprocessing)
        labeled_ds = filenames_ds.map(decode_fn,
                                      num_parallel_calls=AUTOTUNE)

        # Cache the dataset if fits in memory.
        if FLAGS.cache:
            labeled_ds = labeled_ds.cache()

        # Do data augmentation and preprocessing now if it wasn't done before.
        if not do_preprocessing:
            augment_fn = partial(augment_and_preprocess,
                                 size=size)
            labeled_ds = labeled_ds.map(augment_fn,
                                        num_parallel_calls=AUTOTUNE)

        # Shuffle the dataset.
        labeled_ds = labeled_ds.shuffle(buffer_size=1000)

        # Repeat the dataset undefinitely.
        labeled_ds = labeled_ds.repeat()

        # Create batches using buckets: images with similar height will
        # be in the same batch. Minimum extra padding is added if needed.
        # Buckets are intervals of 10 pixels, from 60 to 200.
        bucket_boundaries = [i for i in range(60, 201, 10)]
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        labeled_ds = labeled_ds.apply(
            tf.data.experimental.bucket_by_sequence_length(
                # Use the height of the images to choose the bucket.
                element_length_func=lambda image, label: tf.shape(image)[0],
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padding_values=(
                    # Padding value for images.
                    -1.,
                    # Padding value for labels.
                    False)
            )
        )
        # This will pad the images to the largest size in the batch, since
        # I set target size to None.
        # Labels will not be padded, but it is required to provide shape
        # and value for them too.
        # labeled_ds = labeled_ds.padded_batch(batch_size,
        #                                      padded_shapes=(
        #                                          # Target shape for images.
        #                                          [None, None, 3],
        #                                          # Target shape for labels.
        #                                          [len(class_names)]
        #                                      ),
        #                                      padding_values=(
        #                                          # Padding value for images.
        #                                          -1.,
        #                                          # Padding value for labels.
        #                                          False))
        labeled_ds = labeled_ds.prefetch(buffer_size=AUTOTUNE)

        return labeled_ds, np.array(class_names)

    else:
        print("Cannot load {} split, directory not found!".format(split))
