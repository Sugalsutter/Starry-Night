import numpy as np
from PIL import Image

# Returns a numpy array representing an image.
# Input: string filename of image.
def load_image(filename):
    with Image.open(filename) as img:
        return np.asarray(img)

# Returns a numpy array of numpy arrays representing a
# list of images.
# Input: array of strings of filenames for images.
def load_images(filenames):
    image_data = []
    for f in filenames:
        print(f)
        with Image.open(f) as img:
            image_data.append(np.asarray(img))
    return np.asarray(image_data)
