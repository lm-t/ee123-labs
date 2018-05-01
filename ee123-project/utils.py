""" Utility methods for the project
        - loading images
        - ...
"""
from PIL import Image
import numpy as np

def load_img(image):
    try:
        return Image.open(image)
    except IOError:
        return None
