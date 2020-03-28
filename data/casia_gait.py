from absl import app, flags
from absl.flags import FLAGS
import os
import cv2

flags.DEFINE_string("dataset_dir", "casia_gait/DatasetB-1",
                    "Path to original CASIA Gait dataset")
flags.DEFINE_string("output_dir", "casia_gait/DatasetB_split",
                    "Target directory to save the new dataset")

def split_video(video_path, output_path):
    # Take the video name and cut .avi
    video_name = os.path.basename(video_path)[:-4]

    frame_num = 1
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        # Image file name is video name followed by 3-digit frame number
        img_name = "{}-{:03}.jpg".format(video_name, frame_num)
        img_path = os.path.join(output_path, img_name)
        cv2.imwrite(img_path, image)
        success, image = vidcap.read()
        frame_num += 1

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

def main(_argv):
    video_dir = os.path.join(FLAGS.dataset_dir, "video")
    for file_name in ["001-bkgrd-000.avi"]:  # os.listdir(video_dir):
        dest = get_file_dest(file_name)
        if dest is not None:
            video_path = os.path.join(video_dir, file_name)
            output_path = os.path.join(FLAGS.output_dir, dest)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            split_video(video_path, output_path)


if __name__ == "__main__":
    app.run(main)
