#!/bin/bash

data_path="/scratch/izar/ckli/rendered_jiff_complete"
mask_path="MASK"

# List all folders in data_path+mask_path, sort them, and store only the names of the first 100 folders in used.txt
find "$data_path/$mask_path" -maxdepth 1 -type d -printf "%f\n" | sort | head -n 100 > "$data_path/used.txt"

# List the names of all folders in data_path+mask_path excluding the MASK folder, sort them, and store the last 20 names in val.txt
find "$data_path/$mask_path" -maxdepth 1 -type d -not -name "$mask_path" -printf "%f\n" | sort | tail -n 20 > "$data_path/val.txt"
