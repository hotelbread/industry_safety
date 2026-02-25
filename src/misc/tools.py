import os
import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage


def makeSaveDir(path):
    os.makedirs(path, exist_ok=True)
     
def get_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ---------------------------------------------------
# Image

# Read
def read_img(img_path):
    img_array = np.fromfile(img_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

# Resize
def resize_img(img, target_w, target_h):
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

# Get qimage format
def get_qimg_format(img):
    qimg_format = None
    if img.shape[2] == 3:
        qimg_format = QImage.Format_RGB888
    elif img.shape[2] == 4:
        qimg_format = QImage.Format_RGBA8888
    else:
        raise ValueError("Unsupported image format.")
    return qimg_format

# Convert Pixmap
def convert_BGR2RGB(img):
    rgb_img = None
    if img.shape[2] == 3:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported image format.")
    return rgb_img

# Convert Pixmap
def convert_to_pixmap(img, qimg_format):
    h, w, _ = img.shape
    keyframe_qImg = QImage(img.copy(), w, h, img.strides[0], qimg_format)
    return QPixmap.fromImage(keyframe_qImg)

# Aspect ratio
def get_aspect_ratio(img):
    ori_h, ori_w = img.shape[:2]
    return ori_w / ori_h

# ---------------------------------------------------------
# Resize
# Convert Pixmap
def resize_and_convert_to_pixmap(img, width, height):
    resized_img = resize_img(img, width, height)
    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)

# Resize (based W)
# Convert Pixmap
def resized_pixmap_based_w(img, width):
    aspect_ratio = get_aspect_ratio(img)
    height = int(width / aspect_ratio)
    resized_img = resize_img(img, width, height)

    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)

# Resize (based H)
# Convert Pixmap
def resized_pixmap_based_h(img, height):
    aspect_ratio = get_aspect_ratio(img)
    width = int(height * aspect_ratio)
    resized_img = resize_img(img, width, height)

    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)


# Read
# Resize
# Convert Pixmap
def get_resized_pixmap(img_path, width, height):
    img = read_img(img_path)
    resized_img = resize_img(img, width, height)
    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)

# Read
# Resize (based W)
# Convert Pixmap
def get_resized_pixmap_based_w(img_path, width):
    img = read_img(img_path)

    aspect_ratio = get_aspect_ratio(img)
    height = int(width / aspect_ratio)
    resized_img = resize_img(img, width, height)

    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)\

# Read
# Resize (based H)
# Convert Pixmap
def get_resized_pixmap_based_h(img_path, height):
    img = read_img(img_path)

    aspect_ratio = get_aspect_ratio(img)
    width = int(height * aspect_ratio)
    resized_img = resize_img(img, width, height)

    rgb_resized_img = convert_BGR2RGB(resized_img)
    qimg_format = get_qimg_format(rgb_resized_img)
    return convert_to_pixmap(rgb_resized_img, qimg_format)