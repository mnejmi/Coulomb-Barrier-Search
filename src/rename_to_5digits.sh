#!/bin/bash

set -euo pipefail

TARGET_DIR="/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90"

cd "$TARGET_DIR" || {
    echo " ERROR: Cannot access $TARGET_DIR"
    exit 1
}

echo " Working directory: $(pwd)"
echo " Previewing renames..."

for f in data_2d_*.pt; do
    raw=$(echo "$f" | sed -E 's/data_2d_([0-9]+)\.pt/\1/')
    num=$((10#$raw))   # force base-10

    if [ ${#raw} -lt 5 ]; then
        new=$(printf "data_2d_%05d.pt" "$num")
        echo "  $f  ->  $new"
    fi
done

echo
read -p "️  Proceed with these renames? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo " Aborted."
    exit 0
fi

echo "️  Renaming files..."
for f in data_2d_*.pt; do
    raw=$(echo "$f" | sed -E 's/data_2d_([0-9]+)\.pt/\1/')
    num=$((10#$raw))

    if [ ${#raw} -lt 5 ]; then
        new=$(printf "data_2d_%05d.pt" "$num")
        mv "$f" "$new"
    fi
done

echo " Done. All filenames are now correctly padded to 5 digits."