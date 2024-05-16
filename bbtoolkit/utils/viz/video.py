import cv2
import os
import re


def make_video_from_images(image_dir: str, save_path: str, fps: int = 30):
    """Creates and saves an MP4 video from a set of PNG images in a directory.

    This function reads PNG images from a specified directory, sorts them based on the numeric value in their filenames, and compiles them into an MP4 video at a specified frame rate.

    Args:
        image_dir (str): The directory containing the PNG images.
        save_path (str): The path where the MP4 video will be saved.
        fps (int, optional): Frames per second for the output video. Defaults to 30.

    Returns:
        None
    """
    # Get all PNG files in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]

    # Sort images by the frame number in their name
    images.sort(key=lambda x: int(re.findall("(\d+)", x)[-1]))

    # Read the first image to get the video dimensions
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_dir, image))
        out.write(frame)  # Write out frame to video

    out.release()
    logging.info(f"Video saved to {save_path}")