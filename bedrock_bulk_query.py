import boto3
import csv
import json
import os

from datetime import datetime
from loguru import logger as log
from pathlib import Path

# Local imports
from bedrock_utils import get_foundation_model_ids, invoke_model, get_model_invoke_body


# CSV file with questions
CSV_FILE_NAME = "bedrock-bulk-questions.csv"

# Ask for confirmation before querying each model
ASK_FOR_CONFIRMATION = False # True

client = boto3.client("bedrock")
client_runtime = boto3.client("bedrock-runtime")


def get_bedrock_fm_questions() -> list:
    questions = list([])
    with open(CSV_FILE_NAME, newline="") as file:
        reader = csv.reader(file)
        # Skip the first row (headers)
        next(reader)
        # Filter column results to only get the questions
        questions = [q[0] for q in list(reader)]

    if not questions:
        log.error(f"Failed to read {CSV_FILE_NAME}")
        exit(1)

    log.info(f"Questions: {questions}")
    return questions


def get_fm_query_results(date: str, model_id: str, fm_questions: list) -> dict:
    model_results = dict(
        {
            "model_id": model_id,
            "path_to_file": f"./results/{date}/{model_id}-prompt-results.csv",
            "results": list([]),
        }
    )

    for question in fm_questions:
        query_response = query_fm(question, model_id)
        model_results["results"].append(
            {"model_id": model_id, "question": question, "response": query_response}
        )

    return model_results


def query_fm(prompt: str, model_id: str) -> str:
    """Query a foundation model with a prompt"""
    fm_invoke_body = get_model_invoke_body(model_id, prompt)
    log.info(f"Querying Model ID: {model_id} - Invoke Body: {fm_invoke_body}")
    response = invoke_model(client_runtime, model_id, fm_invoke_body)

    if response is None:
        return "No response from Amazon Bedrock"

    return response


def convert_fm_results_to_csv(results: list) -> list:
    """Converts a list of results to a csv file"""
    csv_results = list([])
    for result in results:
        csv_results.append([result["question"], result["response"]])
    return csv_results


def generate_reports(artifact: dict, json_file: str):
    if not Path(json_file).parent.exists():
        Path(json_file).parent.mkdir(parents=True)

    # Write full JSON results
    with open(json_file, "w") as json_file:
        json.dump(artifact, json_file, indent=2)

    # Write a CSV file for each model
    csv_headers = ["model_id", "question", "response"]
    for model_results in artifact["results"]:
        csv_file = model_results["path_to_file"]
        with open(csv_file, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(model_results["results"])


def main():
    log.info("Testing Amazon Bedrock FM responses with bulk questions")

    script_name = os.path.basename(__file__)
    script_to_file_name = script_name.replace(".py", "").replace("_", "-")
    date = datetime.now().strftime("%Y-%m-%d")

    artifact = dict(
        {"metadata": {"name": script_name, "date": date}, "results": list([])}
    )

    fm_questions = get_bedrock_fm_questions()
    model_ids = get_foundation_model_ids(client)

    for model_id in model_ids:
        log.info(f"Model ID: {model_id}")
        if ASK_FOR_CONFIRMATION:
          confirm = input("Continue? (y/n): ")
          if confirm != "y":
              log.info("Exiting...")
              break

        query_results = get_fm_query_results(date, model_id, fm_questions)
        artifact["results"].append(query_results)

    log.info(f"Artifact: {artifact}")
    json_file = f"./results/{date}/{script_to_file_name}-script-results.json"
    generate_reports(artifact, json_file)


if __name__ == "__main__":
    main()
