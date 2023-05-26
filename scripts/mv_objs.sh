need_list=('0318' '0139' '0134' '0500' '0496' '0135' '0497' '0136' '0498' '0137' '0144' '0499' '0492' '0493' '0132' '0494' '0133' '0140' '0480' '0495' '0141' '0142' '0143' '0344' '0390')

for i in "${need_list[@]}"; do
    source_file="/scratch/izar/ckli/thuman_aligned/scans/$i/$i.obj"
    destination_file="/scratch/izar/ckli/rendered_jiff_complete/GEO/OBJ/$i"
    cp "$source_file" "$destination_file"
    echo "Copied $source_file to $destination_file"
done
