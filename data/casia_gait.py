from absl import app, flags
from absl.flags import FLAGS
import os
import cv2
from zipfile import ZipFile
import tarfile

flags.DEFINE_string("dataset_dir", "casia_gait/DatasetB-1",
                    "Path to original CASIA Gait dataset")
flags.DEFINE_string("output_dir", "casia_gait/DatasetB_split",
                    "Target directory to save the new dataset")
flags.DEFINE_string("zip_pwd", None,
                    "Dataset zip password, if necessary")


# Extracts and deletes all .zip and .tar.gz files inside a path
def extract_and_delete(path):
    for file_name in os.listdir(path):
        if file_name.endswith(".zip"):
            archive_path = os.path.join(path, file_name)
            pwd = bytes(FLAGS.zip_pwd, "utf-8")
            with ZipFile(archive_path, "r") as archive:
                archive.extractall(path=path, pwd=pwd)
            os.remove(archive_path)
        elif file_name.endswith(".tar.gz"):
            archive_path = os.path.join(path, file_name)
            archive = tarfile.open(archive_path)
            archive.extractall(path=path)
            os.remove(archive_path)


# Returns "train/subject_id", "val/subject_id", "test/subject_id" or None,
# which tells what is the destination directory of the file.
def get_file_dest(file_name):
    dest = None

    if file_name.endswith(".avi"):
        # Walking status can be "nm" (normal), "cl" (in a coat),
        # "bg" (with a bag) or "bkgrd" (background).
        subject_id, walk_status, seq_num, *_ = file_name.split("-")
        if walk_status == "nm":
            if seq_num == "01":
                dest = "train"
        elif walk_status == "cl" or walk_status == "bg":
            if seq_num == "01":
                dest = "val"
            elif seq_num == "02":
                dest = "test"

    if dest is not None:
        dest = os.path.join(dest, subject_id)

    return dest


# Splits a video into frames and applies masks to get silhouettes
def split_video(video_path, dest_dir, silhouettes_dir):
    # Take the video name and cut .avi
    video_name = os.path.basename(video_path)[:-4]

    frame_num = 1
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        # Image file name is video name followed by 3-digit frame number
        image_name = "{}-{:03}".format(video_name, frame_num)

        image_masked = apply_mask(image, image_name, silhouettes_dir)

        image_path = os.path.join(dest_dir, image_name + ".jpg")
        if image_masked is not None and not os.path.exists(image_path):
            print("Writing image")
            cv2.imwrite(image_path, image_masked)

        success, image = vidcap.read()
        frame_num += 1


# Applies the .png mask to an image, to obtain the silhouette
def apply_mask(image, image_name, silhouettes_dir):
    subject_id, walk_status, seq_num, view_angle, *_ = \
        image_name.split("-")

    mask_dir = os.path.join(
        silhouettes_dir,
        subject_id,
        walk_status + "-" + seq_num,
        view_angle)
    mask_name = image_name + ".png"
    mask_path = os.path.join(mask_dir, mask_name)

    masked_image = None
    # Not all images have their mask
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Find the bbox around the mask
        bbox = cv2.boundingRect(mask)
        x, y, w, h = bbox
        # Crop the mask using the bbox coordinates
        mask = mask[y:y+h+1, x:x+w+1]
        # Crop the image using the bbox coordinates
        image = image[y:y+h+1, x:x+w+1]

        masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Return None if image doesn't have its mask
    return masked_image


def main(_argv):
    # Extract dataset if it is not already extracted
    parent_dir = os.path.abspath(os.path.join(FLAGS.dataset_dir, os.pardir))
    extract_and_delete(parent_dir)

    # Extract siilhouettes if they are not already extracted
    silhouettes_dir = os.path.join(FLAGS.dataset_dir, "silhouettes")
    extract_and_delete(silhouettes_dir)

    video_dir = os.path.join(FLAGS.dataset_dir, "video")
    for file_name in os.listdir(video_dir):
        dest = get_file_dest(file_name)
        if dest is not None:
            video_path = os.path.join(video_dir, file_name)
            dest_dir = os.path.join(FLAGS.output_dir, dest)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            print("Splitting", file_name)
            split_video(video_path, dest_dir, silhouettes_dir)


if __name__ == "__main__":
    app.run(main)
