from PIL import Image

def resize_image(input_image_path, output_image_path, target_size=(160, 160)):
    """
    Resize the input image to the target size and save it to the output path.
    
    Parameters:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the resized image.
        target_size (tuple): Target size for the resized image in the format (width, height).
    """
    # Open the input image
    with Image.open(input_image_path) as img:
        # Resize the image
        resized_img = img.resize(target_size)
        # Save the resized image
        resized_img.save(output_image_path)

# Example usage:
input_image_path = "./global1.png"
output_image_path = "resized_image.png"

# Resize the input image and save it to the output path
resize_image(input_image_path, output_image_path)
