from absl import app, flags
from absl.flags import FLAGS
import os
import cv2
import tarfile

flags.DEFINE_string("dataset_dir", "casia_gait/DatasetB-1",
                    "Path to original CASIA Gait dataset")
flags.DEFINE_string("output_dir", "casia_gait/DatasetB_split",
                    "Target directory to save the new dataset")


# Extracts and deletes all .tar.gz files inside a path
def extract_and_delete(path):
    for file_name in os.listdir(path):
        if file_name.endswith(".tar.gz"):
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
            dest = "train"
        elif walk_status == "cl" or walk_status == "bg":
            if seq_num == "01":
                dest = "val"
            elif seq_num == "02":
                dest = "test"

    if dest is not None:
        dest = os.path.join(dest, subject_id)

    return dest


def split_video(video_path, dest_dir):
    # Take the video name and cut .avi
    video_name = os.path.basename(video_path)[:-4]

    frame_num = 1
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        # Image file name is video name followed by 3-digit frame number
        img_name = "{}-{:03}.jpg".format(video_name, frame_num)
        img_path = os.path.join(dest_dir, img_name)
        cv2.imwrite(img_path, image)
        success, image = vidcap.read()
        frame_num += 1


# Applies .png masks to full images, to obtain silhouettes
def apply_masks(images_dir, silhouettes_dir):
    for file_name in os.listdir(images_dir):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(images_dir, file_name)
            image_name, *_ = file_name.split(".")

            subject_id, walk_status, seq_num, view_angle, *_ = \
                image_name.split("-")

            mask_dir = os.path.join(
                silhouettes_dir,
                subject_id,
                walk_status + "-" + seq_num,
                view_angle)
            mask_name = image_name + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            # Not all images have their mask
            if os.path.exists(mask_path):
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                masked_image = cv2.bitwise_and(image, image, mask=mask)

                cv2.imwrite(image_path, masked_image)
            # Delete images that don't have their mask
            else:
                os.remove(image_path)


def main(_argv):
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

            split_video(video_path, dest_dir)

            apply_masks(dest_dir, silhouettes_dir)


if __name__ == "__main__":
    app.run(main)
