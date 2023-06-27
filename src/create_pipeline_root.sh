#!/bin/bash

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -s|--source-dir)
      source_dir="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get current timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create a destination directory with gs://gbi_ml/gbi_classification_pipeline/pipeline_runs/<timestamp>
destination_dir="gs://public-projects/demo/ecommerce/${timestamp}"

# Check if source directory is empty or not provided
if [ -z "${source_dir}" ] || [ -z "$(ls -A "${source_dir}")" ]; then
  # gsutil cp command requires a file to be present in the source directory
  # One way to create an empty file is to create an empty text file 
  # and upload it to the destination directory
  
  # Create an empty text file
  echo "Creating an empty text file."
  touch empty.txt

  # Upload the empty text file to the destination directory
  echo "Uploading the empty text file to ${destination_dir}"
  gsutil cp empty.txt "${destination_dir}/empty.txt"

  # Delete the empty text file
  echo "Deleting the empty text file."
  rm empty.txt
else
  # Copy the data to destination directory from the source directory
  echo "Copying data from ${source_dir} to ${destination_dir}"
  gsutil cp -r "${source_dir}" "${destination_dir}"
fi

# Print the destination directory for copy
echo "Data uploaded to ${destination_dir}, contents:"
gsutil ls -al $destination_dir
echo "Created Pipeline Root: ${destination_dir}"
echo "Done."
