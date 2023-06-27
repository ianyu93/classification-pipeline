import kfp
from kfp import compiler, dsl

# Load components from kfp generated YAML files
data_preparation_op = kfp.components.load_component_from_file(
    "components/data_preparation/component_metadata/data_preparation.yaml"
)

training_op = kfp.components.load_component_from_file(
    "components/training/component_metadata/training.yaml"
)


# Define pipeline
@dsl.pipeline(name="huggingface-classification-training-pipeline")
def training_pipeline(
    project_id: str,
    pipeline_root: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    label_mapping_fp: str,
    model_name: str,
    model_dir: str,
    pretrained_model: str = "distilbert-base-uncased",
    learning_rate: float = 1e-5,
    epochs: int = 3,
    batch_size: int = 16,
    eval_steps: int = 1000,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.001,
    deploy_compute: str = "n1-standard-8",
    deploy_accelerator: str = "NVIDIA_TESLA_P100",
    deploy_accelerator_count: int = 1,
):
    """Trains a Huggingface squence classification model on the given train,
    validation, and test datasets. The model is saved to the given model
    directory, and requires label mapping to be provided. The model is then
    registered to Vertex AI and evaluated against the test dataset. The
    evaluation results are then finally logged to Vertex AI.

    Args:
        project_id: Project ID of the project to run the pipeline in.
        pipeline_root: GCS path to store pipeline artifacts.
        train_csv: GCS path to the training dataset.
        val_csv: GCS path to the validation dataset.
        test_csv: GCS path to the test dataset.
        label_mapping_fp: GCS path to the label mapping file.
        model_name: Name of the model to be registered to Vertex AI.
            Make a descriptive name that is easy to identify.
        model_dir: GCS path to store the trained model artifacts.
        pretrained_model: Pretrained model to use for training. Must be
            available in the Huggingface model hub. Defaults to
            "distilbert-base-uncased".
        learning_rate: Learning rate for the optimizer. Defaults to 1e-5.
        epochs: Number of epochs to train for. Defaults to 3.
        batch_size: Batch size for training. Defaults to 16.
        eval_steps: Number of steps to evaluate the model on the validation
            dataset. Defaults to 1000.
        gradient_accumulation_steps: Number of gradient accumulation steps.
            The larger the gradient accumulation steps, the more steps it takes
            before the optimizer updates the model weights. Defaults to 1.
        early_stopping_patience: Number of epochs to wait before stopping
            training if the validation metrics does not improve. Defaults to 3.
        early_stopping_threshold: Threshold for the validation metrics to
            improve by before the early stopping patience is reset. Defaults to
            0.001 to represent 0.1%.
        deploy_compute: Compute machine type used for training. Defaults to
            "n1-standard-8".
        deploy_accelerator: Accelerator type used for training. Defaults to
            "NVIDIA_TESLA_P100".
        deploy_accelerator_count: Number of accelerators used for training.
            Defaults to 1.

    Returns:
        None
    """
    import os
    import uuid

    from google_cloud_pipeline_components.experimental.evaluation import (
        ModelEvaluationClassificationOp, ModelImportEvaluationOp)
    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.batch_predict_job import \
        ModelBatchPredictOp
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from kfp.components import importer_node

    # Batch prediction has to take in string as input to the gcs_source_uris
    # parameter, as the path has to be known at compile time, not at runtime.
    # Both PipelineParam object and Kubeflow objects would not be accepted.
    # Therefore, we need to create a unique ID for each pipeline run and define
    # the gcs_source_uris parameter at compile time rather than runtime
    # Prepare data from CSV to JSONL
    unique_id = str(uuid.uuid4())
    data_preparation_output_dir = (
        "gs://public-projects/demo/ecommerce/runs/data_preparation"
    )
    data_prepartion_output_file = os.path.join(
        data_preparation_output_dir, unique_id, "data.jsonl"
    )

    # Prepare data from CSV to JSONL
    data_preparation_task = data_preparation_op(
        input_file=test_csv,
        output_file=data_prepartion_output_file,
    )
    # Train model
    custom_training_task = training_op(
        pretrained_model=pretrained_model,
        train_set=train_csv,
        val_set=val_csv,
        model_dir=model_dir,
        label_mapping=label_mapping_fp,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )

    # Upload model to Vertex AI
    unmanaged_model_importer = importer_node.importer(
        artifact_uri=model_dir,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-central1-docker.pkg.dev/independent-bay-388105/public/huggingface-sequence-classification:latest",
                "predictRoute": "/predict",
                "healthRoute": "/health",
            }
        },
    )

    model_upload_task = ModelUploadOp(
        project=project_id,
        display_name=f"{model_name}",
        unmanaged_container_model=unmanaged_model_importer.outputs["artifact"],
    )
    model_upload_task.after(custom_training_task)

    # Batch prediction
    batch_predict_op = ModelBatchPredictOp(
        project=project_id,
        job_display_name="batch_predict_job",
        model=model_upload_task.outputs["model"],
        gcs_source_uris=[data_prepartion_output_file],
        gcs_destination_output_uri_prefix=pipeline_root,
        instances_format="jsonl",
        predictions_format="jsonl",
        model_parameters={},
        machine_type=deploy_compute,
        starting_replica_count=1,
        max_replica_count=1,
        accelerator_type=deploy_accelerator,
        accelerator_count=deploy_accelerator_count,
    ).after(model_upload_task, data_preparation_task)

    # ModelEvaluationClassificationOp requires the class labels to be passed in
    # as a list. We can't pass in a PipelineParam object as the class labels
    # parameter. Therefore, we need to create a component to read the class
    # labels from the label_mapping_fp file and return it as a list.
    @dsl.component(
        packages_to_install=["fsspec==2022.11.0", "gcsfs==2022.11.0"]
    )
    def get_label_mapping(label_mapping_fp: str) -> list:
        """Get the class labels from the label_mapping_fp file.

        Args:
            label_mapping_fp: The path to the label mapping file.

        Returns:
            A list of class labels.
        """
        import json

        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(label_mapping_fp) as f:
            label_mapping = json.load(f)
        return list(label_mapping.values())

    classes = get_label_mapping(label_mapping_fp=label_mapping_fp).output

    # Evaluate model by comparing the predictions with the ground truth
    # and generate classification metrics to be used for model evaluation.
    eval_task = ModelEvaluationClassificationOp(
        project=project_id,
        model=model_upload_task.outputs["model"],
        predictions_gcs_source=batch_predict_op.outputs["gcs_output_directory"],
        class_labels=classes,
        prediction_score_column="prediction.scores",
        prediction_label_column="prediction.labels",
        ground_truth_gcs_source=[data_prepartion_output_file],
        target_field_name="label",
        dataflow_machine_type="n1-standard-8",
    )

    # Import model evaluation metrics object to Vertex AI
    # TODO: Investigate where ModelImportEvaluationOp lives in GCPC >=2.0.0
    ModelImportEvaluationOp(
        classification_metrics=eval_task.outputs["evaluation_metrics"],
        model=model_upload_task.outputs["model"],
        dataset_type="jsonl",
    ).after(eval_task)


if __name__ == "__main__":
    # Compile the pipeline to generate the YAML file, reusable on Vertex AI
    compiler.Compiler().compile(
        training_pipeline, package_path="training_pipeline.yaml"
    )
    
    # Compile the pipeline to generate the JSON file, reusable programmatically
    compiler.Compiler().compile(
        training_pipeline, package_path="training_pipeline.json"
    )