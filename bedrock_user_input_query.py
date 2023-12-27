"""
Simple script to test the Amazon Bedrock Boto3 client

Call with `python test.py`
"""
import boto3
import json
import os

from loguru import logger as log

# Local imports
from bedrock_utils import get_foundation_model_ids, invoke_model, get_model_invoke_body


client = boto3.client("bedrock")
client_runtime = boto3.client("bedrock-runtime")

log.info("Amazon Bedrock client created")


def get_user_model_selection(fm_ids: list[str]) -> str:
    log.info("Please select an option by entering its number:")
    for i, option in enumerate(fm_ids, start=1):
        log.info(f"{i}. {option}")
    # Take user input
    max_choice = len(fm_ids)
    try:
        user_choice = int(input(f"Enter your choice (1-{max_choice}): "))
        # Process the input and check if it's a valid choice
        if 1 <= user_choice <= len(fm_ids):
            return fm_ids[user_choice - 1]
        else:
            log.warning(
                f"Invalid choice. Please enter a number between 1 and {max_choice}"
            )
            exit(1)
    except ValueError:
        log.error("Invalid input. Please enter a number.")
        exit(1)


def main():
    log.info("Starting chatbot")
    log.info(f"Python version: {os.sys.version}")
    log.info(f"Pip install location: {os.path.dirname(boto3.__file__)}")
    log.info(f"Boto3 version: {boto3.__version__}")

    fm_ids = get_foundation_model_ids(client)
    log.info(f"Foundation Model IDs: {fm_ids}")

    # TBD since models have different body types
    model_id = get_user_model_selection(fm_ids)
    log.info(f"Model ID: {model_id}")
    prompt = input("Enter your prompt: ")
    log.info(f"Prompt: {prompt}")

    invoke_body = get_model_invoke_body(model_id, prompt)
    log.info(f"Querying Amazon Bedrock - Model: {model_id} Message: '{invoke_body}'")

    response = invoke_model(client_runtime, model_id, invoke_body)
    log.info(f"Response from Amazon Bedrock: '{response}'")
    if response is None:
        return "No response from Amazon Bedrock"

    return response


if __name__ == "__main__":
    main()
