#!/bin/bash

folders=(
   
    
)


for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    python extract_data.py --folder_path "$folder"
    if [ $? -eq 0 ]; then
        echo "Successfully processed $folder"
    else
        echo "Error occurred while processing $folder"
    fi
done
