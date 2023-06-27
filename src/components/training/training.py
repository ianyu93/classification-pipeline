from kfp import dsl

# Custom Training


@dsl.component(
    base_image="huggingface/transformers-pytorch-gpu:4.23.1",
    packages_to_install=[
        "protobuf==3.20.1",
        "fsspec==2022.11.0",
        "gcsfs==2022.11.0",
        "google-cloud-secret-manager",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "scikit-learn",
        "pandas==1.5.2",
        "evaluate",
    ],
    target_image="us-central1-docker.pkg.dev/independent-bay-388105/public/"
    "training",
)
def training(
    train_set: str,
    val_set: str,
    model_dir: str,
    label_mapping: str,
    pretrained_model: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    eval_steps: int,
    gradient_accumulation_steps: int,
    early_stopping_patience: int,
    early_stopping_threshold: float,
):
    """Trains a Huggingface squence classification model on the given train and
    validation datasets. The model is saved to the given model directory, and
    requires label mapping to be provided.

    There are a number of parameters for Huggingface trainer, which is detailed
    on Confluence: https://bit.ly/41uFGcE

    Args:
        train_set: Train dataset path on GCS in CSV format.
        val_set: Validation dataset path on GCS in CSV format.
        model_dir: Model directory path on GCS.
        label_mapping: Label mapping path on GCS in JSON format.
        pretrained_model: Pretrained model name on Huggingface model hub.
        learning_rate: Learning rate for training.
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        eval_steps: Number of steps to run evaluation.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        early_stopping_patience: Number of epochs to wait before early stopping.
        early_stopping_threshold: Threshold for early stopping.

    Returns:
        None
    """
    import json
    import logging

    import evaluate
    import gcsfs
    import numpy as np
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        IntervalStrategy,
        Trainer,
        TrainingArguments,
    )

    # Create Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Formatting handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Use accuracy as metrics for now, can add more sophistication to
    # compute_metrics later
    target_metric = evaluate.load("accuracy")
    CHECKPOINT = pretrained_model

    def create_datasets(ds, tokenizer):
        """Tokenizes all datasets in ds using tokenizer, and returns train and
        validation datasets as a tuple.

        Args:
            ds: Huggingface DatasetDict object containing train and validation
                datasets.
            tokenizer: Huggingface tokenizer object.

        Returns:
            Tuple of train and validation datasets.
        """
        tokenized_datasets = ds.map(
            (lambda examples: tokenize_function(examples, tokenizer)),
            batched=True,
        )

        # To speed up training, we use only a portion of the data.
        # Use full_train_dataset and full_eval_dataset if you want to train on
        # all the data.
        return tokenized_datasets["train"], tokenized_datasets["validation"]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return target_metric.compute(predictions=predictions, references=labels)

    def tokenize_function(examples, tokenizer):
        """Tokenizes text examples."""

        return tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    fs = gcsfs.GCSFileSystem()

    # Prepare label mapping
    logging.info(f"Label mapping path: {label_mapping}")
    with fs.open(label_mapping, "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

    # Prepare model. ignore_mismatched_sizes is set to True to avoid error when
    # loading model from checkpoint. This is used to remove head layer of
    # another trained sequence classifier.

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Prepare datasets
    with fs.open(train_set, "r") as f:
        logging.info(f"Train set path: {train_set}")
        train_df = pd.read_csv(f)[["text", "label"]]
        train_df = train_df[train_df["text"].notna()]
    with fs.open(val_set, "r") as f:
        logging.info(f"Validation set path: {val_set}")
        val_df = pd.read_csv(f)[["text", "label"]]
        val_df = val_df[val_df["text"].notna()]

    # Convert labels to ids for training purposes
    train_df["label"] = train_df["label"].apply(lambda x: label2id[x])
    val_df["label"] = val_df["label"].apply(lambda x: label2id[x])
    logging.info("Label mapping: {}".format(id2label))
    logging.info("Train set: {}".format(train_df))
    logging.info("Validation set: {}".format(val_df))

    # Prepare Huggingface Datasets
    # Often on Huggingface forum, you'd see tds and vds instead of
    # train_dataset and val_dataset

    train_dataset_untokenized = Dataset.from_pandas(train_df)
    val_dataset_untokenized = Dataset.from_pandas(val_df)
    dataset = DatasetDict()

    dataset["train"] = train_dataset_untokenized
    dataset["validation"] = val_dataset_untokenized

    # Create train and validation set
    # This tokenizes datasets using tokenizer and returns train and validation
    # datasets as a tuple
    train_ds, val_ds = create_datasets(dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./model_output",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS,
        lr_scheduler_type="linear",
        log_level="debug",
        logging_strategy="steps",
        save_total_limit=3,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        ],
    )

    # Train model
    trainer.train()

    # Save model locally first, then copy model artifacts direct
    trainer.save_model("./model_output")
    fs.put("./model_output/", model_dir, recursive=True)
