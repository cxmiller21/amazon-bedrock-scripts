# Amazon Bedrock Python API Demo Project

Use the Amazon Bedrock FMs to generate responses to user input.

## Pre-requisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Pip](https://pip.pypa.io/en/stable/installing/)
- [AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
- [Amazon Bedrock FM access](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)

## Getting Started

1. Clone or fork this repo and `cd` into the project directory
   ```bash
   git clone https://github.com/cxmiller21/aws-bedrock-scripts.git
   cd aws-bedrock-scripts
   ```
2. Install dependencies
   ```bash
    python -m pip install -r requirements.txt
    ```
3. (Optional) Modify the bedrock-bulk-questions.csv file
4. Run the scripts
   ```bash
   # Single Bedrock FM query
   python bedrock_user_input_query.py

   # Multiple Bedrock FM queries
    python bedrock_bulk_query.py
   ```
