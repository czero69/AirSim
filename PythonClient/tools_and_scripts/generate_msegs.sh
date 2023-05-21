#!/bin/bash

# my_path="/mnt/d/Kamil/data_collected/airsim_drone/dataset_coffing"
my_path="/mnt/d/Kamil/data_collected/first_vids_matrix_like_real"

for dir_name in "$my_path"/*; do
    if [ -d "$dir_name" ]; then
        input_dir="$dir_name/screenshot"
        resized_dir="./tmp/$(basename $dir_name)"
        output_file="$dir_name/resized_output.jpg"
        model_name="mseg-3m"
        model_path="/mnt/q/KAMIL/myEPE/mseg-3m.pth"
        save_folder="$dir_name/msegs"
        # config="/mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/config/test/1080/default_config_batched_ss.yaml"
        config="/mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/config/test/default_config_1080_ms.yaml"

        # Create the directories if they don't exist
        mkdir -p "$resized_dir"
        mkdir -p "$save_folder"

        # Resize images in the input_dir to 1920x1080

        for img_file in "$input_dir"/*; do
            if [ -f "$img_file" ]; then
                img_name=$(basename "$img_file")
                resized_file="$resized_dir/$img_name"
                echo "Input File: $img_file"
                echo "Resized File: $resized_file"
                convert "$img_file" -resize 1920x1080\! "$resized_file"
            fi
        done

        # unfortunately crash after 15-18 images, cuda outof mem, so process in smaller chunks
        #python -u /mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/tool/universal_demo.py \
        #    --config="$config" model_name "$model_name" model_path "$model_path" input_file "$resized_dir" save_folder "$save_folder"


        resized_dir_tmp="$resized_dir/chunk_10"
        mkdir -p "$resized_dir_tmp"  # Create temporary directory if it doesn't exist

        files_processed=0

        for file in "$resized_dir"/*; do
            mv "$file" "$resized_dir_tmp"  # Move file to temporary directory
            files_processed=$((files_processed + 1))

            if [ "$files_processed" -eq 10 ]; then
                # Run the Python command on the temporary directory which has no more than 10 files
                python -u /mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/tool/universal_demo.py \
                    --config="$config" model_name "$model_name" model_path "$model_path" \
                    input_file "$resized_dir_tmp" save_folder "$save_folder"

                rm -rf "$resized_dir_tmp"/*  # Delete the contents of the temporary directory
                files_processed=0
            fi
        done

        # Process the remaining files in the temporary directory if any
        if [ "$files_processed" -gt 0 ]; then
            python -u /mnt/q/KAMIL/dependencies/mseg-semantic/mseg_semantic/tool/universal_demo.py \
                --config="$config" model_name "$model_name" model_path "$model_path" \
                input_file "$resized_dir_tmp" save_folder "$save_folder"

            rm -rf "$resized_dir_tmp"/*  # Delete the contents of the temporary directory
        fi
        # Remove the resized images after processing
        rm -r "$resized_dir"
    fi
done