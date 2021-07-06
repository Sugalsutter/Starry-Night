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

# Class handles preparing batches of (test_data, target) pairs.
class Loader():

    # Constructor
    #
    # Input
    # image_data (string): directory containing JPEG image files
    # solutions_file (string): CSV file containing training data solutions
    def __init__(self, image_dir, solutions_file):
        self.image_dir = image_dir
        self.solutions_file = solutions_files

    # Returns a batch of (test_data, target) pairs of a given size.
    #
    # Input
    # size (integer): the number of (test_data, target) pairs to return
    #
    # Returns array of (test_data, target) pairs.
    def get_batch(size):
        pass
