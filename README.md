- [About The Project](#about-the-project)
- [Quickstart](#quickstart)
- [How to Run Pipelines](#how-to-run-pipelines)
  - [Components](#components)
  - [Compile Pipelines](#compile-pipelines)
    - [Create Pipeline Root and Upload Data](#create-pipeline-root-and-upload-data)
  - [Run Pipelines](#run-pipelines)
    - [Pipeline Parameters in YAML File](#pipeline-parameters-in-yaml-file)
    - [Pipeline Parameters in JSON File](#pipeline-parameters-in-json-file)
    - [Pipeline Parameters in JSON String](#pipeline-parameters-in-json-string)
  - [Upload Pipelines](#upload-pipelines)
    - [Pipeline Template for Vertex AI](#pipeline-template-for-vertex-ai)
    - [Pipeline Template for Programmatic Runs](#pipeline-template-for-programmatic-runs)
- [Custom Prediction Routines](#custom-prediction-routines)
  - [What is CPR?](#what-is-cpr)
  - [CPR for This Project](#cpr-for-this-project)
- [Roadmap](#roadmap)
  - [Training Pipeline Model Evaluation Import](#training-pipeline-model-evaluation-import)
  - [Other Pipeline Implementation](#other-pipeline-implementation)
  - [Cloud Run API](#cloud-run-api)

# About The Project

This project is a demonstration of Kubeflow usage with Google Cloud Pipeline Components. It focuses on packaging and deploying programmatically rather than through Jupyter Notebook to reflect on production needs.

# Quickstart

This quickstart will guide you through the process of training and deploying a model.

First, please make sure you have `train.csv`, `val.csv`, `test.csv`, and `label_mapping.json` under `data/` folder. Each of `train.csv`, `val.csv`, and `test.csv` should have "id", "text", "label" columns, and `label_mapping.json` should have a mapping from integer inext to labels, without skipping any integer. In the format of:

```JSON
{
    "0": "label_0",
    "1": "label_1",
    ...
}
```

Then, create a pipeline root and upload data to the pipeline root by running the following command:

```bash
bash create_pipeline_root.sh -s data/
```

This will output a GCS path on the terminal, which is the pipeline root, which looks like:

```
Created Pipeline Root: gs://public-projects/demo/ecommerce/20230626_143552
```

Copy the pipeline root. Then create a `pipeline_param.yaml`, see [Available Pipelines](src/README.md#available-pipelines) for various YAML templates.

```bash
touch pipeline_param.yaml
```

Here is the template for training pipeline, replace <PIPELINE_ROOT> with the pipeline root you just copied, and <MODEL_NAME> with the model name you want to use:

```yaml
# pipeline_param.yaml
project_id: independent-bay-388105
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

Finally, run the pipeline by running the following command:

```bash
python3 run.py -p independent-bay-388105 -l "us-central1" -c "gs://public-projects/templates/training_pipeline.json" -r <PIPELINE_ROOT> -d "Training Pipeline" -ppy "pipeline_param.yaml" -s "1079975490159-compute@developer.gserviceaccount.com"
```

# How to Run Pipelines

In this project, we have implemented containerized components and compiled pipelines under `src/`.

## Components

We have defined containerized components under `src/components`, where each Python script contains a single component with a specified GCR image path, for example, the following snippet is an implementation of the `data_preparation` component, where the target image path is `gcr.io/groupby-development/gb-ml/classification-pipeline/data-preparation`:

```python
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "gcsfs==2022.11.0",
        "google-cloud-secret-manager",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "jsonlines",
        "pandas==1.5.2",
        "protobuf==3.20.*",
    ],
    target_image="us-central1-docker.pkg.dev/independent-bay-388105/public/"
    "data-preparation",
)
def data_preparation(input_file: str, output_file: str):
    ...
```

To build and push component images to GCR, run the following command:

```bash
cd src/components
bash create_image.sh
```

`create_image.sh` will build and push all components to GCR.

Under `src/components/<component>/component_metadata`, YAML files for each built containerized component are available to be loaded for the pipeline implementation.

## Compile Pipelines

After the components are built and pushed to GCR, we can compile the pipelines under `src/`. Each pipeline is defined in a Python script. Run Python command to compile the pipeline, for example, the following command compiles the `inference_pipeline.py` pipeline:

```bash
cd src
python src/training_pipeline.py
```

This will create both `training_pipeline.yaml` and `training_pipeline.json`. `training_pipeline.yaml` could be stored on Artifact Registry's Kubeflow Pipeline repository, which allows the pipeline template to be run on Vertex AI via GUI. `training_pipeline` contains definition of the pipeline and is used for submitting pipeline runs programmatically, via `run.py` for this project.

### Create Pipeline Root and Upload Data

As pipelines require a GCS path to store pipeline artifacts, we need to create a GCS path and optionally upload data to the bucket. For this purpose, we have `src/create_pipeline_root.sh` to create a GCS pipeline root. If you already have a pipeline root, then you can skip ahead to [Run Pipelines](#run-pipelines).

`src/create_pipeline_root.sh` would take the current timestamp, and create a GCS path with the format of:

`gs://public-projects/demo/ecommerce/${timestamp}`

While it's not possible to create an empty folder via `gsutil`, the script has a workaround to create an empty file locally, upload the file to the pipeline root, and then delete the file. This is to ensure that the pipeline root is created and can be used for pipeline runs.

To create a pipeline root, run the following command:

```bash
bash src/create_pipeline_root.sh
```

`src/create_pipeline_root.sh` can optionally take in a source directory and copy all files in the directory to the pipeline root. For example, the following command creates a pipeline root and copies all files in `data/` to the pipeline root:

```bash
# both -s or --source-directory would work
bash src/create_pipeline_root.sh -s data/
```

The script would still upload a `empty.txt` file if the source directory specified is empty.

The script would output the pipeline root path via the terminal, which can be used as pipeline parameters.

## Run Pipelines

To run a specific pipeline, you would need to first [compile](#compile-pipelines) a pipeline and have a pipeline JSON file, and then submit the compiled pipeline JSON file to `run.py`.

Here are the arguments for `run.py`:

- `-p/--project-id`: the ID of the Google Cloud project that the pipeline will be run on.
- `-l/--location`: the location where the pipeline will be run, in the format of a GCP region (e.g. `us-central1`).
- `-d/--display-name`: the display name of the pipeline job.
- `-c/--compiled-pipeline-path`: the path to the compiled pipeline file, could be a local path or a GCS path.
- `-r/--pipeline-root`: the GCS path where the pipeline outputs will be stored.
- `-s/--service-account`: the service account to use for running the pipeline job.
- `-ppy/--pipeline-parameters-yaml` (optional): the path to a YAML file containing the pipeline parameters to be passed to the pipeline.
- `-ppj/--pipeline-parameters-json` (optional): the path to a JSON file containing the pipeline parameters to be passed to the pipeline.
- `-pps/--pipeline-parameters-json_string` (optional): a JSON string containing the pipeline parameters to be passed to the pipeline.

For pipeline parameters, you can either pass them in a YAML file, a JSON file, or a JSON string. 

### Pipeline Parameters in YAML File

If you pass them in a YAML file, the YAML file should be in the following format:

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

And then run:

```bash
python3 run.py -p "independent-bay-388105" -l "us-central1" -c "gs://public-projects/templates/training_pipeline.json" -r "<PIPELINE_ROOT>" -d "Training Pipeline" -ppy "pipeline_param.yaml" -s "1079975490159-compute@developer.gserviceaccount.com"
```

### Pipeline Parameters in JSON File

If you define parameters in a JSON file, the JSON file should be in the following format:

```json
// pipeline_param.json
{
    "project_id": "independent-bay-388105",
    "pipeline_root": "<PIPELINE_ROOT>",
    "train_csv": "<PIPELINE_ROOT>/train.csv",
    "val_csv": "<PIPELINE_ROOT>/val.csv",
    "test_csv": "<PIPELINE_ROOT>/test.csv",
    "label_mapping_fp": "<PIPELINE_ROOT>/label_mapping.json",
    "model_name": "<MODEL_NAME>",
    "model_dir": "<PIPELINE_ROOT>/model",
    "pretrained_model": "distilbert-base-uncased",
    "learning_rate": 1e-5,
    "epochs": 3,
    "batch_size": 16,
    "eval_steps": 1000,
    "gradient_accumulation_steps": 1,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    "deploy_compute": "n1-standard-8",
    "deploy_accelerator": "NVIDIA_TESLA_P100",
    "deploy_accelerator_count": 1
}

```

And then run:

```bash
python3 run.py -p "independent-bay-388105" -l "us-central1" -c "gs://public-projects/templates/training_pipeline.json" -r "<PIPELINE_ROOT>" -d "Training Pipeline" -ppj "pipeline_param.yaml" -s "1079975490159-compute@developer.gserviceaccount.com""
```

### Pipeline Parameters in JSON String

If you pass them as a JSON string, as often the case with requests, run:

```bash
python3 run.py \
-p "independent-bay-388105" \
-l "us-central1" \
-c "gs://public-projects/templates/training_pipeline.json" \
-r "<PIPELINE_ROOT>" \
-d "Training Pipeline" \
-pps '{"project_id": "independent-bay-388105", \
"pipeline_root": "<PIPELINE_ROOT>", \
"train_csv": "<PIPELINE_ROOT>/train.csv", \
"val_csv": "<PIPELINE_ROOT>/val.csv", \
"test_csv": "<PIPELINE_ROOT>/test.csv", \
"label_mapping_fp": "<PIPELINE_ROOT>/label_mapping.json", \
"model_name": "<MODEL_NAME>", \
"model_dir": "<PIPELINE_ROOT>/model", \
"pretrained_model": "distilbert-base-uncased", \
"learning_rate": 1e-5, \
"epochs": 3, \
"batch_size": 16, \
"eval_steps": 1000, \
"gradient_accumulation_steps": 1, \
"early_stopping_patience": 3, \
"early_stopping_threshold": 0.001, \
"deploy_compute": "n1-standard-8", \
"deploy_accelerator": "NVIDIA_TESLA_P100", \
"deploy_accelerator_count": 1, \
"output_format": "jsonl"}' \
 -s "1079975490159-compute@developer.gserviceaccount.com"
```

## Upload Pipelines

As mentioned in [Compile Pipelines](#compile-pipelines), `python src/<pipeline>.py` would create both `<pipeline>.yaml` and `<pipeline>.json`. 

### Pipeline Template for Vertex AI
`<pipeline>.yaml` could be stored on Artifact Registry's Kubeflow Pipeline repository, which allows the pipeline template to be run on Vertex AI via GUI. For more details about how to upload a pipeline template for Vertex AI, please refer to [this page](https://cloud.google.com/vertex-ai/docs/pipelines/create-pipeline-template).

For this project, `src/upload_pipeline.py` is a script that uploads a pipeline template to a preset Artifact Registry's Kubeflow Pipeline repository. In order to upload `<pipeline>.yaml` to the repository, run:   
```bash
cd src
python upload_pipeline.py file_name --tag tag_name --desription description
```


### Pipeline Template for Programmatic Runs
 `<pipeline>.json` contains definition of the pipeline and is used for submitting pipeline runs programmatically, via `run.py` for this project. 

 For this project, we simply upload `<pipeline>.json` to a GCS bucket, and then use the GCS path to submit pipeline runs. This requires authentication to the Google Cloud Project. See how to authorize the gcloud CLI [here](https://cloud.google.com/sdk/docs/authorizing).

Assuming you are authenticated with `gcloud`, you can run the following command to upload `<pipeline>.json` to a GCS bucket:

`gsutil cp <pipeline>.json gs://<bucket_name>/templates/`

For this project, it would be `gs://public-projects/templates/<pipeline>.json` Then, you can use the GCS path to submit pipeline runs programmatically, via `run.py` for this project.

# Custom Prediction Routines

For this project, we have a custom training component that creates a Huggingface Sequence Classifciation, where the model artifacts are stored on GCS. Whether we want to deploy it for online prediction or for batch prediction, we would need to create a [Custom Prediction Routine](https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines#:~:text=Custom%20prediction%20routines%20(CPR)%20lets,building%20a%20container%20from%20scratch.), abbreviated as CPR.

## What is CPR?
A custom serving container contains the following 3 pieces of code:

1. [Model Server](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/model_server.py) 
   - HTTP server that hosts the model.
   - Responsible for setting up routes/ports/etc.
2. [Request Handler](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/handler.py)
   - Responsible for webserver aspects of handling a request, such as deserializing the request body, serializing the response, setting response headers, etc.
   - Default Handler is google.cloud.aiplatform.prediction.handler.PredictionHandler provided in the SDK.
3. [Predictor](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/predictor.py)
   - Responsible for the ML logic for processing a prediction request.

With the CPR image, it can also be deployed as a Model Endpoint on Vertex AI, which is basically a serving container that can be used to serve predictions. When the model endpoint is being requested, `artifact_url` will be supplied, and `PredictionHandler` will be execute the following:
   
   ```python
      self._predictor.postprocess(self._predictor.predict(self._predictor.preprocess(prediction_input)))
   ```
In the above execution, the Predictor will be downloading content from `artifact_url` and load the model artifacts. Then, the Predictor will be executing the prediction logic, and finally, the Predictor will be returning the prediction result.

## CPR for This Project
For this project, we have implemented custom prediction routine under `src/prediction_pipeline`. `src/prediction_pipeline` is the source code directory that contains the custom serving container code to handle the prediction requests. This includes a custom predictor, but one can also have custom handler and customer server. 

Create a `requirements.txt` file that contains the dependencies for the custom serving container. In this case, the dependencies are the Hugging Face Transformers library and the Google Cloud AI Platform SDK.

We have `src/cpr.py` to build and push the custom prediction routine to GCR. To build and push the custom prediction routine to GCR, run the following command:

```bash
cd src
python cpr.py
```

The registered image would be able to take in a `artifact_url` parameter, which is the GCS path to the model artifacts, to perform batch prediction or online prediction. 

As a result of training pipeline, a remote `model_artifacts` directory is created on GCS, which contains the model artifacts. This will be passed as the parameter for `artifact_url`. For this project, the model artifacts are the Hugging Face Transformers model, and the `artifact_url` is used to perform batch prediction in `training_pipeline`. 

# Roadmap

This project is still under development. As it was a bare reimplementation of a previous project, there are still some features unimplemented and bugs to be investigated.

The previous project was developed before Kubeflow V2 and Google Cloud Pipeline Components V2 were released as stable versions. We want to keep the project up to date with the latest Kubeflow and Google Cloud Pipeline Components, as there are additional features from Kubeflow V2 highly desirable for production use cases.

## Training Pipeline Model Evaluation Import
In the training pipeline, `ModelImportEvaluationOp` is used to evaluate the model. However, `ModelImportEvaluationOp` was a legacy artifact left from Google Cloud Pipeline Components before V2. Currently, there isn't documentation on how to adapt `ModelImportEvaluationOp` on V2, which will require further investigation.

## Other Pipeline Implementation
There are other pipeline implementation that could be done. For example:
- Dynamic training pipeline, where the pipeline would be able to serialize a list of pretrained model names and train them all via [kfp.dsl.ParallelFor](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.ParallelFor) control flow.
- Dynamic inference pipeline, where the pipeline would be able to serialize a list of model resource names on Vertex AI and perform batch prediction
- Ensemble probability aggregation, where the pipeline would be able to serialize a list of model resource names on Vertex AI and perform batch prediction, and then aggregate the probabilities from each model to produce a final prediction via [kfp.dsl.Collected](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.Collected) control flow, only available on Kubeflow V2.


## Cloud Run API
In the previous project, a Cloud Run service was deployed to allow users to make requests and run pipelines without having to use the Google Cloud SDK. This would allow development of self-serve GUI for users to run pipelines. For example, for stakeholders to test predict models on their custom data.

`src/run.py` is designed to accept JSON string precisely for this purpose, where the Cloud Run service would take JSON payload from request and resend the payload to `src/run.py` on the service.

Although the Cloud Run service is not deployed, the implementation is available under `src/api`.