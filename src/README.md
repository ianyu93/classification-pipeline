# Available Pipelines

Here we lay out available pipelines, their usage, arguments, and expected outputs.

- For how to compile a pipeline, please see [Compile Pipelines](../README.md#compile-pipelines).

- For how to create a pipeline root on GCS, please see [Create Pipeline Root](../README.md#create-pipeline-root-and-upload-data)

- For how to run a pipeline, please see [Run Pipelines](../README.md#run-pipelines).

## Table of Contents

- [Available Pipelines](#available-pipelines)
  - [Table of Contents](#table-of-contents)
- [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [YAML Template](#yaml-template)

# Training and Evaluation Pipeline

## Usage

This pipeline trains a Huggingface sequence classification model on the provided train, validation, and test datasets. The trained model is saved to the specified model directory and registered to Google Cloud's Vertex AI. The evaluation results are logged to Vertex AI.

The pipeline consists of the following steps:

1. **Model Training**: The pipeline trains a sequence classification model using the provided train and validation datasets. It utilizes a pretrained model available in the Huggingface model hub (default: "distilbert-base-uncased").

2. **Model Registration**: Once trained, the model is registered to Vertex AI with a user-specified name. This step allows you to easily deploy and serve the trained model for making predictions.

3. **Model Evaluation**: The registered model is evaluated against the test dataset. Evaluation metrics, such as accuracy, precision, and recall, are computed to assess the model's performance.

4. **Logging Results**: The evaluation results are logged to Vertex AI, providing a centralized location to monitor and track model performance.

## Arguments

Here are the arguments to be specified when running the pipeline:

- `project_id`: This is your Google Cloud project ID. It is a unique identifier for your project.

- `pipeline_root`: This is the GCS (Google Cloud Storage) path to store pipeline artifacts. This is where the pipeline will store any temporary files or output files.

- `train_csv`: The GCS path to the training dataset. This file contains the training examples for the model. It should be in CSV format, with "id" and "text" columns, optionally with a "label" column.

- `val_csv`: The GCS path to the validation dataset. This file contains examples used for model validation during training. It should be in CSV format, with "id" and "text" columns, optionally with a "label" column.

- `test_csv`: The GCS path to the test dataset. This file contains examples used to evaluate the final model's performance. It should be in CSV format, with "id" and "text" columns, optionally with a "label" column.

- `label_mapping_fp`: The GCS path to the label mapping file. This file maps label indices to human-readable labels. This would be model's prediction output. It should be in JSON format, with the label index as the key and the label as the value.

- `model_name`: The name of the model to be registered in Vertex AI. Choose a descriptive name that makes it easy to identify the model.

- `model_dir`: The GCS path to store the trained model artifacts. This is where the pipeline will save the trained model files. Typically, it's a subdirectory of the pipeline root.

- `pretrained_model`: The pretrained model to use for training. It should be available in the Huggingface model hub. Default model is `distilbert-base-uncased` for efficiency.

- `learning_rate`: The learning rate for the optimizer during training, defaulting to 1e-5.

- `epochs`: The number of epochs to train the model. An epoch represents one complete pass through the training data. The default number of epochs is 3, as transformers models typically converge quickly over batches and may overfit with more epochs.

- `batch_size`: The batch size for training. It defines the number of examples processed together during each training step. The default batch size is 16 to ensure memory efficiency. If you run into memory issues, you can try reducing the batch size.

- `eval_steps`: The number of steps to evaluate the model on the validation dataset. It determines how frequently the model's performance is evaluated during training. The default number of steps is 1000, but you can increase this number to evaluate the model more frequently.

- `gradient_accumulation_steps`: The number of gradient accumulation steps. Larger values delay the optimizer's weight updates, potentially improving training stability. Default value is 1 to update the weights after each batch.

- `early_stopping_patience`: The number of epochs to wait before stopping training if the validation metrics do not improve. Default value is 3.

- `early_stopping_threshold`: The threshold for the validation metrics to improve by before resetting the early stopping patience. Default value is 0.001 to represent a 0.1% improvement.

- `deploy_compute`: This is the machine type for batch prediction. You will need to specify the machine type that you want to use. The available options depend on the region you are working in. For Huggingface models, the recommended machine type is at least `n1-standard-8`.

- `deploy_accelerator`: This is the accelerator type for batch prediction. You will need to specify the accelerator type that you want to use, if any. The available options depend on the machine type you choose. For this project, the recommended accelerator type is at least `NVIDIA_TESLA_P100`.

- `deploy_accelerator_count`: This is the number of accelerators for batch prediction. You will need to specify the number of accelerators that you want to use, if any. The available options depend on the accelerator type you choose. For this project, the recommended number of accelerators is at least `1`.

## YAML Template

```yaml
# pipeline_param.yaml
project_id: "independent-bay-388105"
pipeline_root: "<PIPELINE_ROOT>"
train_csv: "<PIPELINE_ROOT>/train.csv"
val_csv: "<PIPELINE_ROOT>/val.csv"
test_csv: "<PIPELINE_ROOT>/test.csv"
label_mapping_fp: "<PIPELINE_ROOT>/label_mapping.json"
model_name: "<MODEL_NAME>"
model_dir: "<PIPELINE_ROOT>/model"
pretrained_model: "distilbert-base-uncased"
learning_rate: 1e-5
epochs: 3
batch_size: 16
eval_steps: 1000
gradient_accumulation_steps: 1
early_stopping_patience: 3
early_stopping_threshold: 0.001
deploy_compute: "n1-standard-8"
deploy_accelerator: "NVIDIA_TESLA_P100"
deploy_accelerator_count: 1
```