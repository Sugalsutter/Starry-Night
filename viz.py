import matplotlib.pyplot as plt
from PIL import Image

# Display image.
# Input: string filename of image.
def show_image(filename):
    with Image.open(filename) as img:
        plt.imshow(img)
        plt.show()
