import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

RUNPOD_API_URL = "https://api.runpod.ai/v2/c4qbicuam59iw1"
RUNPOD_API_KEY = os.getenv("RUNPOD_APIKEY", "")


def runpod_request(endpoint: str, input_data: dict, sync: bool = True) -> dict:
    url = f"{RUNPOD_API_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": input_data}
    if sync:
        payload["synchronous"] = True

    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def runpod_async_request(endpoint: str, input_data: dict) -> str:
    url = f"{RUNPOD_API_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": input_data}

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("id", "")


def runpod_status(job_id: str) -> dict:
    url = f"{RUNPOD_API_URL}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def runpod_stream(job_id: str):
    url = f"{RUNPOD_API_URL}/stream/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.get(url, headers=headers, stream=True, timeout=300)
    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    yield {"raw": line[6:]}


def extract_text(text: str, use_llm: bool = False) -> dict:
    input_data = {
        "text": text,
        "use_llm": use_llm,
    }
    return runpod_request("extract", input_data, sync=True)


def extract_text_async(text: str, use_llm: bool = False) -> str:
    input_data = {
        "text": text,
        "use_llm": use_llm,
    }
    return runpod_async_request("extract", input_data)
