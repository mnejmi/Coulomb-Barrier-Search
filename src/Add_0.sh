#!/bin/bash

TARGET_DIR="/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90"

cd "$TARGET_DIR" || exit 1

echo "Starting to rename files..."

for i in {1..1383}; do
    old_num=$(printf "%04d" "$i")
    new_num=$(printf "%05d" "$i")
    
    old_file="data_2d_${old_num}.pt"
    new_file="data_2d_${new_num}.pt"
    
    if [ -f "$old_file" ]; then
        mv "$old_file" "$new_file"
    fi
done

echo "Renaming completely finished!"