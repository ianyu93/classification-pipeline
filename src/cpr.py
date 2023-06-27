import os

from google.cloud.aiplatform.prediction import LocalModel

from prediction_pipeline.predictor import CustomPredictor

USER_SRC_DIR = "prediction_pipeline"
PROJECT_ID = "independent-bay-388105"
REPOSITORY = "public"
IMAGE = "huggingface-sequence-classification"
USER_SRC_DIR = "prediction_pipeline"

local_model = LocalModel.build_cpr_model(
    USER_SRC_DIR,  # source code, Docker will copy this directory to the image
    output_image_uri=f"us-central1-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}:latest",  # where on GCS to store the image
    base_image="python:3.9",  # base image to use
    predictor=CustomPredictor,  # custom predictor class
    requirements_path=os.path.join(
        USER_SRC_DIR, "requirements.txt"
    ),  # requirements.txt file in the source code directory
)

# To check out container spec
local_model.get_serving_container_spec()
local_model.push_image()
