"""Usage of this script:
python -m apps.prepare_corruption <directory_path>
directory_path: The path to the directory containing the images to be corrupted. Example usage:
python -m apps.prepare_corruption /scratch/izar/ckli/corruptions_benchmark/final
"""
import os
import sys
import random


def save_names_to_file(directory):
    all_subjects = set()

    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):
            obj_num = file_name.split('_')[-3]
            all_subjects.add(obj_num)
    print(all_subjects)
    sampled_subjects = sorted(list(all_subjects))[:5]
    
    # Save the sampled names into a text file
    output_file = 'bench_val.txt'
    save_path = os.path.join(directory, output_file)
    with open(save_path, 'w') as f:
        for subject in sampled_subjects:
            f.write(subject + '\n')

    print(f"Randomly sampled names saved to {save_path}.")

# Check if directory argument is provided
if len(sys.argv) < 2:
    print("Please provide the directory path as an argument.")
else:
    directory_path = sys.argv[1]
    save_names_to_file(directory_path)
