import json
import os
import subprocess

from flask import Flask, request

app = Flask(__name__)


@app.route("/run_pipeline", methods=["POST", "PUT"])
def run_pipeline():
    """Runs a predefined pipeline template by making use of the provided
    parameters in a POST or PUT request.

    This function extracts parameters from the incoming JSON request, forms a
    command, and uses subprocess to run the command.

    Parameters in the JSON should include:
    - project_id: The ID of the Google Project where the pipeline will be run.
    - location: The location where the pipeline will be run.
    - compiled_pipeline_path: The path to the compiled pipeline file.
    - pipeline_root: The root directory for the pipeline.
    - display_name: The name to display for the pipeline.
    - pipeline_parameters_json_string: A JSON string of pipeline parameters.
    - service_account: The service account to use when running the pipeline.

    The result of the pipeline execution returns the requester along with a
    HTTP 201 status code indicating successful pipeline submission.

    Note that if there's an issue running the subprocess (e.g., the command
    fails to execute), this function will return an error.

    Returns:
        A tuple containing the standard output from the pipeline execution,
        which includes the created pipeline run URL, and the HTTP status code.
    """
    # Get JSON data from request
    json_data = request.get_json(force=True)

    # Create command to run
    command = [
        "python3",
        "run.py",
        "--project-id",
        json_data["project_id"],
        "--location",
        json_data["location"],
        "--compiled-pipeline-path",
        json_data["compiled_pipeline_path"],
        "--pipeline-root",
        json_data["pipeline_root"],
        "--display-name",
        json_data["display_name"],
        "--pipeline-parameters-json-string",
        json.dumps(json_data["pipeline_parameters_json_string"]),
        "--service-account",
        json_data["service_account"],
    ]

    # Submit the pipeline with subprocess
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # Return the result with a 201 status code and pipeline URL
    return result.stdout, 201


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
