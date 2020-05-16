from absl import app, flags
from absl.flags import FLAGS
import os
import cv2
from zipfile import ZipFile
import tarfile

flags.DEFINE_string("dataset_dir", "casia_gait/DatasetB-1",
                    "Path to original CASIA Gait dataset")
flags.DEFINE_string("output_dir", "casia_gait/generated_video",
                    "Target directory to save the new dataset")


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


# # Returns "train/subject_id", "val/subject_id", "test/subject_id" or None,
# # which tells what is the destination directory of the file.
# def get_file_dest(file_name):
#     dest = None

#     if file_name.endswith(".avi"):
#         # Walking status can be "nm" (normal), "cl" (in a coat),
#         # "bg" (with a bag) or "bkgrd" (background).
#         subject_id, walk_status, seq_num, *_ = file_name.split("-")
#         if walk_status == "nm":
#             if seq_num == "01":
#                 dest = "train"
#         elif walk_status == "cl" or walk_status == "bg":
#             if seq_num == "01":
#                 dest = "val"
#             elif seq_num == "02":
#                 dest = "test"

#     if dest is not None:
#         dest = os.path.join(dest, subject_id)

#     return dest


# Splits a video into frames and applies masks to get silhouettes
def split_video(video_path, silhouettes_dir):
    # Take the video name and cut .avi
    video_name = os.path.basename(video_path)[:-4]

    images = []
    frame_num = 1
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        # Image file name is video name followed by 3-digit frame number
        image_name = "{}-{:03}".format(video_name, frame_num)

        image_masked, coords = apply_mask(image, image_name, silhouettes_dir)

        if image_masked is not None:
            images += [(image_masked, coords)]

        success, image = vidcap.read()
        frame_num += 1
    
    return images

def extract_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    
    return image


# Applies the .png mask to an image, to obtain the silhouette and bbox
# coordinates
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
    x = None
    y = None
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
        
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
        masked_image[:, :, 3] = mask

    # Return None if image doesn't have its mask
    return masked_image, (x, y)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def save_video(images, video_path):
    width = images[0].shape[1]
    height = images[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    for image in images:
        out.write(image)
    
    out.release()


def main(_argv):
    # Extract dataset if it is not already extracted
    parent_dir = os.path.abspath(os.path.join(FLAGS.dataset_dir, os.pardir))
    extract_and_delete(parent_dir)

    # Extract siilhouettes if they are not already extracted
    silhouettes_dir = os.path.join(FLAGS.dataset_dir, "silhouettes")
    extract_and_delete(silhouettes_dir)

    video_dir = os.path.join(FLAGS.dataset_dir, "video")

    background_video = "001-bkgrd-090.avi"
    background_image = extract_frame(os.path.join(video_dir, background_video))

    final_images = [background_image.copy() for _ in range(55)]
    for i, file_name in enumerate(["001-nm-01-090.avi",
                                   "003-nm-01-090.avi",
                                   "009-nm-01-090.avi"]):
        video_path = os.path.join(video_dir, file_name)

        person_images_coords = split_video(video_path, silhouettes_dir)

        for background_image, (person_image, (x, y)) in zip(final_images,
                                                            person_images_coords):
            
            x -= (i * 50)
            overlay_image_alpha(background_image,
                                person_image[:, :, 0:3],
                                (x, y),
                                person_image[:, :, 3] / 255.0)
        
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    dest_path = os.path.join(FLAGS.output_dir, "generated_video.avi")
    save_video(final_images, dest_path)


if __name__ == "__main__":
    app.run(main)
