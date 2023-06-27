#!/bin/bash

# For all directories in the current directory
for COMPONENT_DIR in */; do
  # Skip __pycache__ directory
  if [[ "$COMPONENT_DIR" != "__pycache__/" ]]; then
    # Change into the directory
    cd "$COMPONENT_DIR"

    # For all component python files in the directory
    for COMPONENT in *.py; do
      # KFP builds Docker image and pushes to gcr.io/groupby-development/gbi-ml/COMPONENT_NAME:latest
      # Target image URI is set in each component python file
      kfp components build ./ --component-filepattern "$COMPONENT" --push-image
    done

    # Return to the parent directory
    cd ..
  fi
done
