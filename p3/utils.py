import numpy as np

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    image_min = np.amin(image_data)
    image_max = np.amax(image_data)
    a = 0.1
    b = 0.9
    return a + ( ( (image_data - image_min)*(b - a) )/( image_max - image_min ) )


