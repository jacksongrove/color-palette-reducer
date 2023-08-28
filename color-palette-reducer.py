"""
Color Palette Reducer

This script provides a function to compress the color palette of an image using 
KMeans clustering, creating a unique screen-print-like effect on digital images.
"""

# Imports
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os


# Function to reduce image color palette
def reduce_color_palette(image_path, k=16):
    '''
    Compress the color palette of an image using KMeans clustering.

    This function takes in an image path and a specified number of colors (k). 
    It uses K-Means clustering to identify groups of similar colors in the image 
    and normalizes these groups to their centroid color, effectively reducing the 
    image's color palette.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    k : int, optional
        Number of colors to reduce the image to. Default is 16.

    Returns
    -------
    None
        The function saves the compressed image and displays it.
    '''

    # Open the image file
    img = Image.open(image_path)
    # Convert the image data to a NumPy array
    img_np = np.array(img)

    # Reshape the data to a two-dimensional array where each row is a pixel and the columns are the RGB values
    pixels = img_np.reshape(-1, 3)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    new_colors = kmeans.cluster_centers_

    # Get the labels of each pixel (i.e., which cluster they belong to)
    labels = kmeans.labels_

    # Replace each pixel with the new color (centroid of its cluster)
    compressed_pixels = new_colors[labels].astype(np.uint8)

    # Reshape the compressed pixels to the original image dimensions
    compressed_img_np = compressed_pixels.reshape(img_np.shape)

    # Create a PIL image from the NumPy array
    compressed_img = Image.fromarray(compressed_img_np)

    # Create a new file path for the compressed image
    base_path = os.path.dirname(image_path)
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))
    new_file_path = os.path.join(base_path, f"{file_name}_compressed{file_ext}")

    # Save the image
    compressed_img.save(new_file_path)

    # Display the compressed image
    compressed_img.show()
