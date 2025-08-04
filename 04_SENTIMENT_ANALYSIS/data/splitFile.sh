#!/bin/bash

echo "Creds: GPT"
echo "Usage: ./splitFile.sh input.csv 50m"
input_file="$1"
size_limit="$2"   # e.g., 50m, 100m

if [[ -z "$input_file" || -z "$size_limit" ]]; then
  echo "Usage: $0 <input_file> <size_limit (e.g. 50m)>"
  exit 1
fi

# Get base name and extension
filename=$(basename -- "$input_file")
extension="${filename##*.}"
basename_no_ext="${filename%.*}"

# Split the file (will create files like basename_partaa, etc.)
split -b "$size_limit" -a 2 "$input_file" "${basename_no_ext}_part"

# Rename split files to have .csv or original extension
for file in ${basename_no_ext}_part??; do
  mv "$file" "$file.$extension"
done

echo "Done: Split '$input_file' into parts with .$extension extension"
