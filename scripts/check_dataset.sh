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

