#!/bin/bash
set -e

# Check if a directory path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Navigate to the specified directory
cd "$1" || exit 1

# Iterate over every file in the specified directory
for file in *; do
    # Skip directories
    [ -d "$file" ] && continue
    file_name_without_extension="${file%.txt}"
    # Check if the file has a .txt extension
    if [[ "$file" == *.txt ]]; then
        echo "Running script on file: $file"
        # Execute your script on each file
        cd ../Gn-Glove/GloVe-1.2/
        bash N.sh "$file_name_without_extension" ../../Corpi/"$file" 
        cd ../../Corpi/
    fi
done