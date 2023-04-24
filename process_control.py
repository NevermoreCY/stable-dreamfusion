import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_control(image_path, control_type):
    image = cv2.imread(image_path)

    if control_type == 'canny':
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = image / np.max(image)
        image = np.concatenate([image, image, image], axis=2)

    return image

def preprocess_control_image(image, control_type):

    if control_type == 'canny':
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = image / np.max(image)
        image = np.concatenate([image, image, image], axis=2)

    return image