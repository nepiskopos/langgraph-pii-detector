#!/usr/bin/env python3
"""
Agent Test Script

This script tests the legal entity extraction agent using both synchronous and
asynchronous clients from the langgraph-sdk. It can send multiple test DOCX files
to evaluate extraction quality, response consistency, and server stability.

Usage:
1. Install required packages:
    pip install langgraph-sdk

2. Set your deployment URL and LangSmith API key:
    - Update the DEPLOYMENT_URL and LANGSMITH_API_KEY variables below
    - Or set them as environment variables before running the script

3. Run the script:
    python test_agent.py
"""

import asyncio
import os
import time
import uuid
from langgraph_sdk import get_client, get_sync_client

from filesystem_loader import load_local_documents


async def test_client(files: list, asyncronous: bool=False, threadless: bool=False):
    """Test client connection with one or more input files"""
    print("=== PII Identification and Extraction Agent Test ===")
    print(f"Deployment URL: {DEPLOYMENT_URL}")

    if len (files) == 1:
        print("\n=== Testing Client with One Input File ===")
    else:
        print("\n=== Testing Client with Multiple Input File ===")

    try:
        client = None
        thread_id = None

        if asyncronous:
            print("Using asynchronous client")
            client = get_client(url=DEPLOYMENT_URL, api_key=LANGSMITH_API_KEY)

            print("✅ Asynchronous client created successfully")

            if not threadless:
                thread = await client.threads.create(
                    thread_id=str(uuid.uuid4()),
                )
                thread_id = thread["thread_id"]

            # Get the start time for response time calculation
            start_time = time.time()

            full_response = await client.runs.wait(
                thread_id,
                "agent",    # Name of assistant (defined in langgraph.json)
                input={
                    'files': files,
                },
            )

            # Calculate and display response time
            duration = time.time() - start_time
            print(f"Response time: {duration:.2f} seconds")
            print(100*"=")
            print("Result:", full_response)
            print(100*"=")
        else:
            print("Using synchronous client")
            client = get_sync_client(url=DEPLOYMENT_URL, api_key=LANGSMITH_API_KEY)

            print("✅ Synchronous client created successfully")

            if not threadless:
                thread = client.threads.create(
                    thread_id=str(uuid.uuid4()),
                )
                thread_id = thread["thread_id"]

            # Get the start time for response time calculation
            start_time = time.time()

            full_response = client.runs.wait(
                thread_id,
                "agent",    # Name of assistant (defined in langgraph.json)
                input={
                    'files': files,
                },
            )

            # Calculate and display response time
            duration = time.time() - start_time
            print(f"Response time: {duration:.2f} seconds")
            print(100*"=")
            print("Result:", full_response)
            print(100*"=")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    # Configuration
    # You can set these as environment variables or update them directly here
    LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
    DEPLOYMENT_URL = os.environ.get("DEPLOYMENT_URL", "http://localhost:2024")
    USE_ASYNC_CLIENT = os.environ.get("USE_ASYNC_CLIENT", "true").casefold() == 'true' # Set to False to run synchronously
    USE_THREADLESS_CLIENT = os.environ.get("USE_THREADLESS_CLIENT", "false").casefold() == 'true'  # Set to False to run without threads

    # Test PDF files (add paths to your test files)
    TEST_FILES_DIR = "docx_files"
    TEST_SAMPLE_FILES = [
        # "1.docx",
        # "2.docx",
        # "3.docx",
        # "4.docx",
    ]

    test_files = load_local_documents(TEST_FILES_DIR)

    if TEST_SAMPLE_FILES:
        body_files = [file for file in test_files if file.name in TEST_SAMPLE_FILES]

    if not test_files:
        print("❌ ERROR: No test files found in the specified directory.")

    print(f"Found {len(test_files)} test files")

    # Run the async run_app function
    asyncio.run(test_client(test_files, asyncronous=USE_ASYNC_CLIENT, threadless=USE_THREADLESS_CLIENT))