#!/bin/bash

base_dir="/scratch/izar/ckli/rendered_jiff_complete/GEO/OBJ"
missing_files=()

# Iterate over subdirectories
find "$base_dir" -type d -print0 | while IFS= read -r -d '' dir; do
    # Check if the file exists in the subdirectory
    file_path="$dir/$(basename "$dir").obj"
    if [ ! -f "$file_path" ]; then
        echo "File $file_path does not exist"
        missing_files+=("$(basename "$dir")")
    fi
done

base_directory="/scratch/izar/ckli/rendered_jiff_complete"

for sub_directory in "$base_directory"/*; do
    if [ -d "$sub_directory" ]; then
        if [ "$(basename "$sub_directory")" = "GEO" ]; then
            sub_directory="$sub_directory/OBJ"
        fi

        for i in {0..525}; do
            directory_name=$(printf "%04d" "$i")
            if [ ! -d "$sub_directory/$directory_name" ]; then
                echo "Missing directory: $sub_directory/$directory_name"
            fi
        done
    fi
done

