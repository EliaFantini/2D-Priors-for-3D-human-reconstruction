import shutil

# Define the source and destination directories
source_dir = "/scratch/izar/ckli/thuman_aligned/scans/"
destination_dir = "/scratch/izar/ckli/corruptions_benchmark/all/"

# Define the list of 'num' values
nums = ['0400', '0424', '0312', '0505', '0141', '0294', '0212', '0153', '0379', '0205', '0270', '0122', '0207', '0313', 
        '0427', '0335', '0352', '0350', '0109', '0289', '0299', '0195', '0101', '0386', '0426', '0029', '0318', '0344', 
        '0343', '0118', '0049', '0226', '0489', '0292', '0232', '0031', '0524', '0258', '0399']

# Loop over the 'num' values
for num in nums:
    # Define the source file path
    source_file = source_dir + f"{num}/{num}.obj"
    source_file_mtl = source_dir + f"{num}/{num}.mtl"

    # Define the destination file path
    destination_file = destination_dir + f"{num}.obj"
    destination_file_mtl = destination_dir + f"{num}.mtl"

    # Copy the file from the source to the destination
    shutil.copy2(source_file, destination_file)
    shutil.copy2(source_file_mtl, destination_file_mtl)

print("Files copied successfully.")