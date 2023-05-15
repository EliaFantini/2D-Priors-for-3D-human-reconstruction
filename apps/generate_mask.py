import os
import sys
from PIL import Image
from rembg.session_factory import new_session
from rembg.bg import remove


def process_images(input_directory, output_directory):
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get a list of all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a .png image
        if filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(input_directory, filename)

            # Create a new session for each image
            session = new_session("u2netP")

            # Open the image file
            render = Image.open(file_path).convert('RGB')

            # Run the remove() function
            result = remove(render, alpha_matting=True, session=session)
            result = result.split()[-1]

            # Construct the output file name by appending '_mask' before the '.png' extension
            output_filename = filename[:-4] + '_mask.png'

            # Save the result in the output directory
            result.save(os.path.join(output_directory, output_filename))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    input_directory = input_directory.replace('\\', '/')
    output_directory = sys.argv[2]
    output_directory = output_directory.replace('\\', '/')
    process_images(input_directory, output_directory)
