"""Usage of this script:
python -m apps.prepare_corruption <directory_path>
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
    # Pick the first 5 subjects as the validation set
    # random.seed(42)  # Fix the random seed for reproducibility
    # num_samples = int(len(all_subjects) * 0.7)
    # sampled_subjects = random.sample(all_subjects, num_samples)
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
