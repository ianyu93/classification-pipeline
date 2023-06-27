import argparse

from google.oauth2 import service_account
from kfp.registry import RegistryClient

credentials = service_account.Credentials.from_service_account_file(
    "../service_acc_key.json"
    )
scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])
def upload_pipeline(file_name, tags, description):
    client = RegistryClient(
        host="https://us-central1-kfp.pkg.dev/independent-bay-388105/pipeline-templates",
        auth=scoped_credentials
        )

    templateName, versionName = client.upload_pipeline(
        file_name=file_name,
        tags=tags,
        extra_headers={"description": description}
    )

    return templateName, versionName

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Upload a pipeline to the registry")
    parser.add_argument("file_name", type=str, help="The name of the pipeline YAML file")
    parser.add_argument("--tags", nargs="+", default=[], help="Tags to associate with the pipeline")
    parser.add_argument("--description", type=str, default="", help="Description of the pipeline")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the upload_pipeline function with the provided arguments
    result = upload_pipeline(args.file_name, args.tags, args.description)

    # Print the results
    print("Template Name:", result[0])
    print("Version Name:", result[1])
