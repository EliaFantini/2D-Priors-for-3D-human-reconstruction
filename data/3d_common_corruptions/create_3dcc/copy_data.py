root_folder = "/mnt/c/Users/Elia/Desktop/VIExperimentsMilestone1/thuman2_36views"
destination_path = "/mnt/c/Users/Elia/Desktop/input"
import os
import shutil
# get  a list of all the folders in root_folder and print the first 120
folders = os.listdir(root_folder)
folders.sort()
folders = folders[:20]
print(folders)
for folder in folders:
    # get all the images in the folder and keep only the ones that start with "135", with "225", with "315" and with "45"
    images = os.listdir(os.path.join(root_folder,folder,"render"))
    images = [image for image in images if image.startswith("000") or image.startswith("090") or image.startswith("180") or image.startswith("270")]
    images.sort()
    # get the groundtruth for each image
    for image in images:
        # copy the image to the destination folder using shutil.copy
        if not os.path.exists(os.path.join(destination_path, "groundtruth",folder)):
            os.makedirs(os.path.join(destination_path, "groundtruth",folder))
        shutil.copy(os.path.join(root_folder, folder,"render", image), os.path.join(destination_path, "groundtruth",folder))
    if not os.path.exists(os.path.join(destination_path, "input",folder)):
            os.makedirs(os.path.join(destination_path, "input",folder))
    shutil.copy(os.path.join(root_folder, folder,"render", images[0]), os.path.join(destination_path, "input",folder))

for folder in folders:
    # get all the images in the folder and keep only the ones that start with "135", with "225", with "315" and with "45"
    images = os.listdir(os.path.join(root_folder,folder,"depth_F"))
    images = [image for image in images if image.startswith("000")]
    images.sort()
    # get the groundtruth for each image
    if not os.path.exists(os.path.join(destination_path, "depth",folder)):
            os.makedirs(os.path.join(destination_path, "depth",folder))
    shutil.copy(os.path.join(root_folder, folder, "depth_F",images[0]), os.path.join(destination_path, "depth",folder))

