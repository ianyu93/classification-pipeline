from kfp import dsl


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
    """Prepares the data for training.

    This component is used to prepare the data for training or inference needs.
    It receives an input file in CSV format and outputs a file in JSONL format.
    In the CSV file, there should be the following columns:
    - id: ID of the instance
    - text: Text of the instance
    - label (optional): Label of the instance


    Args:
        input_file: Input file path in GCS bucket in CSV format
        output_file: Output file path in GCS bucket in JSONL format
    """
    import logging

    import gcsfs
    import jsonlines
    import pandas as pd

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

    fs = gcsfs.GCSFileSystem()
    logger.info(f"Input file path: {input_file}")

    with fs.open(input_file, "rb") as f:
        df = pd.read_csv(f)
        logger.info(f"Dataframe shape: {df.shape}")
        logger.info(f"Dataframe columns: {df.columns}")
        logger.info(f"Dataframe head: {df.head()}")

    # ID is forced to be in the schema
    # If Label is not in the schema, it will be empty string
    # Can be repurposed for other use cases
    if "label" not in df.columns:
        df["label"] = ""
    instances = [
        {"sequence": text, "label": bucket_name, "id": id_}
        for text, bucket_name, id_ in zip(
            df["text"].tolist(), df["label"].tolist(), df["id"].tolist()
        )
    ]

    logger.info(f"Number of instances: {len(instances)}")

    # Save the instances to JSONL file on GCS
    with fs.open(output_file, "w") as f:
        writer = jsonlines.Writer(f)
        writer.write_all(instances)
        logger.info(f"Saved to {output_file}")
