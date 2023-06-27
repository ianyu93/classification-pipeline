import argparse
import ast
import json

import yaml
from google.cloud import aiplatform
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "../service_acc_key.json"
    )

def get_args():
    parser = argparse.ArgumentParser(
        prog="Run Pipeline",
        description="Run the pipeline",
    )
    parser.add_argument(
        "-p", "--project-id", required=True, type=str, help="Project ID"
    )
    parser.add_argument(
        "-l", "--location", required=True, type=str, help="Location"
    )
    parser.add_argument(
        "-d", "--display-name", required=True, type=str, help="Display name"
    )
    parser.add_argument(
        "-c",
        "--compiled-pipeline-path",
        required=True,
        type=str,
        help="Compiled pipeline path, could be a GCS path or a local path",
    )
    parser.add_argument(
        "-r", "--pipeline-root", required=True, type=str, help="Pipeline root"
    )
    parser.add_argument(
        "-s",
        "--service-account",
        required=True,
        type=str,
        help="Service account",
    )

    # Pipeline parameters, which can be passed in as a YAML file, JSON file
    # or JSON string
    params = parser.add_mutually_exclusive_group(
        required=True,
    )
    params.add_argument(
        "-ppy",
        "--pipeline-parameters-yaml",
        type=str,
        help="Pipeline parameters YAML file",
    )
    params.add_argument(
        "-ppj",
        "--pipeline-parameters-json",
        type=str,
        help="Pipeline parameters JSON file",
    )
    params.add_argument(
        "-pps",
        "--pipeline-parameters-json-string",
        type=str,
        help="Pipeline parameters JSON string",
    )

    return parser.parse_args()


# Parse arguments
args = get_args()

# Load pipeline parameters from file or string
if args.pipeline_parameters_yaml:
    with open(args.pipeline_parameters_yaml, "r") as f:
        PIPELINE_PARAMETERS = yaml.safe_load(f)
elif args.pipeline_parameters_json:
    with open(args.pipeline_parameters_json, "r") as f:
        PIPELINE_PARAMETERS = json.load(f)
elif args.pipeline_parameters_json_string:
    PIPELINE_PARAMETERS = ast.literal_eval(args.pipeline_parameters_json_string)
else:
    raise ValueError("Pipeline parameters source not specified")

# Submit the pipeline job to Vertex Pipelines
job = aiplatform.PipelineJob(
    display_name=args.display_name,
    template_path=args.compiled_pipeline_path,
    pipeline_root=args.pipeline_root,
    parameter_values=PIPELINE_PARAMETERS,
    project=args.project_id,
    location=args.location,
    credentials=credentials,
)

job.submit(service_account=args.service_account)
