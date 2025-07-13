#!/usr/bin/env python3
"""
Document PII Detection LangGraph Agent Test Script.

This script tests the document PII detection LangGraph agent using either a synchronous
or an asynchronous client from the LangGraph SDK. It can send multiple test documents
to evaluate extraction quality, response consistency, and server stability.

Usage:
1. Install required packages:
   pip install asyncio langchain-community langchain-docling langgraph-sdk pypandoc

2. Set your deployment and local file system paths.

3. Run the script using the command:
    python test_agent.py [OPTIONS]

    Options:
    -a, --address: LangGraph server's IP address (default: 127.0.0.1)
    -p, --port: LangGraph server's serving port (default: 2024)
    -k, --key: LangSmith API access key (optional)
    -s, --synchronous: Use synchronous client (default: False)
    -t, --threadless: Use threadless client execution (default: False)
    -d, --directory: Documents directory path (default: ../documents)
    -f, --files: List of specific files to process (default: all files in the directory)

    Example:
    python test_agent.py -a 127.0.0.1 -p 2024 -k YOUR_API_KEY -s True -d path/to/documents/dir -f file1.pdf file2.docx
"""
from langgraph_sdk import get_client, get_sync_client
from typing import Any, Dict, List
import argparse
import asyncio
import ipaddress
import os
import sys
import time
import uuid

from filesystem_loader import load_local_documents
from logger import get_logger


async def test_client(url: str, port: int, api_key: str, files: List[Dict[str, Any]], syncronous: bool=False, threadless: bool=False):
    """Test client connection with one or more input files"""
    logger.info("=== Document PII Detection Agent Test ===")

    deployment_url = f"http://{url}:{port}"

    logger.info(f"Sending {len(files)} files to the LangGraph deployment URL {deployment_url}:")
    for file in files:
        logger.info(f"- File {file.get('file', {}).get('id', '0')}: {file.get('file', {}).get('filename', 'Unknown')} ({file.get('file', {}).get('meta', {}).get('content_type', 'Unknown')})")

    import json
    print(json.dumps(test_files, indent=2))

    try:
        client = None
        thread_id = None

        if syncronous:
            logger.info("Using synchronous client")

            client = get_sync_client(url=deployment_url, api_key=api_key)

            logger.info("✅ Synchronous client created successfully")

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
            logger.info("✅ Synchronous client completed successfully")
            logger.info(f"Response time: {duration:.2f} seconds")
            logger.info(100*"=")
            logger.info("Result:", full_response)
            logger.info(100*"=")
        else:
            print("Using asynchronous client")

            client = get_client(url=deployment_url, api_key=api_key)

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
            print("✅ Asynchronous client completed successfully")
            print(f"Response time: {duration:.2f} seconds")
            print(100*"=")
            print("Result:", full_response)
            print(100*"=")

    except Exception as e:
        logger.info(f"❌ Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PII Detection LangGraph Agent Test Script')
    parser.add_argument("-a", "--address", help="LangGraph server's IP address", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", help="LangGraph server's serving port", type=int, default=2024)
    parser.add_argument("-k", "--key", help="LangSmith API access key", type=str, required=False)
    parser.add_argument("-s", "--synchronous", help="Use synchronous client", type=bool, default=False)
    parser.add_argument("-t", "--threadless", help="Use threadless client execution", type=bool, default=False)
    parser.add_argument("-d", "--directory", help="Documents directory path", type=str, default="../documents")
    parser.add_argument("-f", "--files", help="List of specific files to process", type=str, nargs='+', default=[])
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.address)
    except ValueError:
        sys.exit(f"Wrong IP address: {args.address}")
    if args.port < 0 or args.port > 65535:
        sys.exit(f"Wrong serving port: {args.port}")
    if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
        sys.exit(f"Wrong path to directory with documents: {args.directory}")

    test_files = load_local_documents(args.directory)

    if args.files:
        test_files = [file for file in test_files if file.get("file", {}).get("filename", "") in args.files]

    if not test_files:
        sys.exit(f"No test files found in directory {args.directory}.")

    logger = get_logger("test_agent")

    logger.info(f"Found {len(test_files)} test files")

    asyncio.run(test_client(args.address, args.port, args.key, test_files, syncronous=args.synchronous, threadless=args.threadless))