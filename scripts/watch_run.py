#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import requests
from requests import RequestException


TERMINAL_STATES = {
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "CANCELED",
    "SKIPPED",
    "ERROR",
}


def load_env_file(env_path: Path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def extract_state(payload: dict):
    run_data = payload.get("run") or payload
    for key in ("state", "status", "phase"):
        value = run_data.get(key)
        if value:
            return str(value).upper()

    pipeline_runtime = run_data.get("pipelineRuntime") or {}
    for key in ("state", "status", "phase"):
        value = pipeline_runtime.get(key)
        if value:
            return str(value).upper()

    return "UNKNOWN"


def watch_run(args):
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    host = (args.host or os.environ.get("KUBEFLOW_HOST") or "http://localhost:8080").rstrip("/")
    url = f"{host}/apis/v2beta1/runs/{args.run_id}"

    print(f"Watching run {args.run_id} at {host}")

    start = time.time()
    last_state = None

    while True:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        state = extract_state(payload)

        if state != last_state:
            elapsed = int(time.time() - start)
            print(f"[{elapsed:>4}s] state={state}")
            last_state = state

        if state in TERMINAL_STATES:
            print(f"Run finished with state={state}")
            if state == "SUCCEEDED":
                return 0
            return 1

        if time.time() - start > args.timeout:
            print(f"Timed out after {args.timeout}s waiting for run {args.run_id}")
            return 124

        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Watch Kubeflow Pipeline run status")
    parser.add_argument("--run-id", required=True, help="KFP run ID")
    parser.add_argument("--host", default=None, help="KFP host (e.g. http://localhost:8080)")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds")
    parser.add_argument("--timeout", type=int, default=1800, help="Max wait time in seconds")
    args = parser.parse_args()

    try:
        raise SystemExit(watch_run(args))
    except KeyboardInterrupt:
        print("Interrupted by user")
        raise SystemExit(130)
    except RequestException as exc:
        print(f"Request failed while watching run: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
