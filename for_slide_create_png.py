import SimpleITK as sitk
import numpy as np
from PIL import Image
import SimpleITK as sitk

def mhd_to_png(input_mhd, output_png):
    # Read the .mhd file using SimpleITK
    image = sitk.ReadImage(input_mhd)

    # Convert the image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Normalize the image array to the range [0, 255]
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    image_array = image_array.astype(np.uint8)

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image_array)

    # Save the PIL image as a PNG file
    pil_image.save(output_png)

# Example usage
input_mhd = 'images_1/original_val1/0.mhd'
output_png = 'output.png'
mhd_to_png(input_mhd, output_png)