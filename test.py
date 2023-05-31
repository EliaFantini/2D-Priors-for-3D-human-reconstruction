import torch
import clip
from PIL import Image
import requests
from torch.nn.functional import cosine_similarity
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

path_list = ['/scratch/izar/ckli/thuman2_36views/0003/render/',
        '/scratch/izar/ckli/thuman2_36views/0027/render/']
output_folder = 'similarities'  # Specify the output folder path

# create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_list_all = []
for angle in range(0, 360, 10):
    name = f'{angle:03d}.png'
    image_path1 = os.path.join(path_list[0], name)
    image_path2 = os.path.join(path_list[1], name)

    with torch.no_grad():
        # Load and preprocess images
        image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
        # rendering: discussions

        # Encode images
        image_features1 = model.encode_image(image1)
        print('image_features1.shape', image_features1.shape)
        image_features2 = model.encode_image(image2)

        # Calculate similarity
        similarity = cosine_similarity(image_features1, image_features2)

    # Plot images side by side and set title with similarity
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(Image.open(image_path1))
    # ax[1].imshow(Image.open(image_path2))
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[0].set_title(f'Similarity: {similarity.item():.4f}')

    # # Save the plot
    # output_path = os.path.join(output_folder, f'plot_{angle}.png')
    # plt.savefig(output_path)
    # plt.close()

    # print(f"Saved plot: {output_path}")
