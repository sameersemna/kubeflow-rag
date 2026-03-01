#!/usr/bin/env python3
"""
Run a Kubeflow RAG pipeline.

Usage:
    python scripts/run_pipeline.py \
        --pipeline pipelines/usecase1_document_qa/pipeline.py \
        --config configs/config.yaml \
        --host https://your-kf-host \
        --experiment my-experiment
"""

import argparse
import importlib.util
import sys
import os
import re
import socket
import ssl
import shlex
import shutil
import subprocess
import warnings
import yaml
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


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


def resolve_env_placeholders(value):
    if not isinstance(value, str):
        return value

    def replace(match):
        var_name = match.group(1)
        default_value = match.group(3)
        env_value = os.environ.get(var_name)
        if env_value:
            return env_value
        if default_value is not None:
            return default_value
        return match.group(0)

    return ENV_PLACEHOLDER_PATTERN.sub(replace, value)


def load_pipeline_module(pipeline_path: str):
    spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = module
    spec.loader.exec_module(module)
    return module


def find_likely_namespace(target_namespace: str, available_namespaces):
    non_system = [
        ns
        for ns in available_namespaces
        if ns not in {"default"} and not ns.startswith("kube-")
    ]

    if not non_system:
        return None

    for ns in non_system:
        if "kubeflow" in ns:
            return ns
    for ns in non_system:
        if "pipeline" in ns or "ml-pipeline" in ns:
            return ns
    for ns in non_system:
        if target_namespace in ns or ns in target_namespace:
            return ns

    return non_system[0]


def preflight_kubeflow_namespace(namespace: str):
    if not namespace:
        return True, None

    if shutil.which("kubectl") is None:
        print("ℹ️  Namespace preflight skipped: kubectl is not installed")
        return True, None

    cmd = ["kubectl", "get", "namespace", namespace, "-o", "name"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except Exception as exc:
        print(f"ℹ️  Namespace preflight skipped: {exc}")
        return True, None

    if result.returncode == 0:
        print(f"✅ Namespace: '{namespace}' exists")
        return True, None

    stderr = (result.stderr or "").strip().splitlines()
    details = stderr[0] if stderr else "namespace check failed"
    print(f"❌ Namespace preflight failed for '{namespace}': {details}")
    print("   Set --namespace or KUBEFLOW_NAMESPACE to a valid namespace")

    list_cmd = ["kubectl", "get", "namespaces", "-o", "jsonpath={.items[*].metadata.name}"]
    list_result = subprocess.run(list_cmd, capture_output=True, text=True)
    if list_result.returncode == 0:
        available = [ns.strip() for ns in list_result.stdout.split() if ns.strip()]
        if available:
            likely_namespace = find_likely_namespace(namespace, available)
            if likely_namespace:
                print(f"💡 Suggested namespace: {likely_namespace}")
                return False, likely_namespace

    return False, None


def preflight_kubeflow_host(host: str, namespace: str):
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        print(f"❌ Invalid Kubeflow host URL: {host}")
        print("   Expected format: https://your-kubeflow-host")
        return False, None

    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"🔎 Preflight: validating Kubeflow host {host}")

    try:
        address_info = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        print(f"❌ Kubeflow host is not resolvable: {parsed.hostname}")
        print("   Check DNS or /etc/hosts, or set KUBEFLOW_HOST to a reachable endpoint")
        return False, None

    print(f"✅ DNS: {parsed.hostname} resolves ({len(address_info)} address records)")

    namespace_ok, suggested_namespace = preflight_kubeflow_namespace(namespace)
    if not namespace_ok:
        return False, suggested_namespace

    tcp_connected = False
    tcp_error = None
    for family, socktype, proto, _, sockaddr in address_info:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(3)
        try:
            sock.connect(sockaddr)
            tcp_connected = True
            break
        except OSError as exc:
            tcp_error = exc
        finally:
            sock.close()

    if not tcp_connected:
        print(f"❌ TCP: cannot connect to {parsed.hostname}:{port}")
        if tcp_error:
            print(f"   Last socket error: {tcp_error}")
        print("   Ensure the Kubeflow endpoint is reachable from this machine/network")
        print_port_forward_suggestions(parsed.hostname, port, namespace)
        return False, None

    print(f"✅ TCP: reachable on {parsed.hostname}:{port}")

    if parsed.scheme == "https":
        tls_ok = False
        tls_error = None
        for family, socktype, proto, _, sockaddr in address_info:
            if socktype != socket.SOCK_STREAM:
                continue
            try:
                with socket.socket(family, socktype, proto) as sock:
                    sock.settimeout(4)
                    sock.connect(sockaddr)
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    with context.wrap_socket(sock, server_hostname=parsed.hostname):
                        tls_ok = True
                        break
            except (ssl.SSLError, OSError) as exc:
                tls_error = exc

        if not tls_ok:
            print(f"❌ TLS: handshake failed for https://{parsed.hostname}:{port}")
            if tls_error:
                print(f"   Last TLS error: {tls_error}")
            print("   Endpoint may not be serving HTTPS on this port")
            return False, None

        print(f"✅ TLS: handshake successful on {parsed.hostname}:{port}")

    base_url = host.rstrip("/")
    health_paths = [
        "/apis/v2beta1/healthz",
        "/pipeline/apis/v2beta1/healthz",
        "/healthz",
    ]

    acceptable_status = {200, 401, 403}
    for path in health_paths:
        probe_url = f"{base_url}{path}"
        request = Request(probe_url, method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                status = getattr(response, "status", 200)
            if status in acceptable_status:
                print(f"✅ HTTP: reachable endpoint {path} (status {status})")
                return True, None
        except HTTPError as exc:
            if exc.code in acceptable_status:
                print(f"✅ HTTP: reachable endpoint {path} (status {exc.code})")
                return True, None
            print(f"ℹ️  HTTP probe {path} returned status {exc.code}")
        except URLError as exc:
            print(f"ℹ️  HTTP probe {path} failed: {exc.reason}")
        except Exception as exc:
            print(f"ℹ️  HTTP probe {path} failed: {exc}")

    print("❌ HTTP: could not verify Kubeflow API health endpoints")
    print("   Confirm ingress URL/path and that Kubeflow Pipelines API is exposed")
    return False, None


def host_looks_like_ui(host: str):
    base_url = host.rstrip("/")
    request = Request(f"{base_url}/", method="GET")
    try:
        with urlopen(request, timeout=4) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type", "")
        if status in {301, 302, 303, 307, 308}:
            return True
        return status == 200 and "text/html" in content_type.lower()
    except HTTPError as exc:
        return exc.code in {301, 302, 303, 307, 308}
    except Exception:
        return False


def resolve_run_view_url(api_host: str, run_id: str, ui_host: str = None):
    if ui_host:
        return f"{ui_host.rstrip('/')}/#/runs/details/{run_id}", None

    if host_looks_like_ui(api_host):
        return f"{api_host.rstrip('/')}/#/runs/details/{run_id}", None

    message = (
        "ℹ️  Host appears to be API-only (UI route not served on this port).\n"
        "   To view runs in browser, port-forward UI and open:\n"
        f"   http://localhost:3000/#/runs/details/{run_id}\n"
        "   Example: kubectl -n kubeflow port-forward svc/ml-pipeline-ui 3000:80"
    )
    return None, message


def build_namespace_retry_command(suggested_namespace: str):
    argv = list(sys.argv)
    if "--namespace" in argv:
        ns_index = argv.index("--namespace")
        if ns_index + 1 < len(argv):
            argv[ns_index + 1] = suggested_namespace
        else:
            argv.append(suggested_namespace)
    else:
        argv.extend(["--namespace", suggested_namespace])

    return "python " + " ".join(shlex.quote(arg) for arg in argv)


def print_port_forward_suggestions(hostname: str, port: int, namespace: str):
    localhost_aliases = {"localhost", "127.0.0.1", "::1"}
    if hostname not in localhost_aliases or port != 8080:
        return

    print("💡 It looks like you are using a local Kubeflow endpoint.")
    print("   If Kubeflow is in-cluster, start a port-forward and re-run this command.")

    if shutil.which("kubectl") is None:
        print("   Suggested command:")
        print("   kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80")
        return

    namespace = namespace or os.environ.get("KUBEFLOW_NAMESPACE", "kubeflow")
    service_candidates = [
        ("ml-pipeline-ui", "8080:80"),
        ("ml-pipeline", "8080:8888"),
    ]

    discovered = []
    for service_name, mapping in service_candidates:
        cmd = ["kubectl", "-n", namespace, "get", "svc", service_name, "-o", "name"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            discovered.append((service_name, mapping))

    if discovered:
        print(f"   Detected services in namespace '{namespace}':")
        for service_name, mapping in discovered:
            print(f"   kubectl -n {namespace} port-forward svc/{service_name} {mapping}")
    else:
        print(f"   Could not auto-detect Kubeflow services in namespace '{namespace}'.")
        print("   Try one of:")
        print(f"   kubectl -n {namespace} port-forward svc/ml-pipeline-ui 8080:80")
        print(f"   kubectl -n {namespace} port-forward svc/ml-pipeline 8080:8888")


def main():
    warnings.filterwarnings(
        "ignore",
        message=r"This client only works with Kubeflow Pipeline v2\.0\.0-beta\.2 and later versions\.",
        category=FutureWarning,
        module=r"kfp\.client\.client",
    )

    parser = argparse.ArgumentParser(description="Run a Kubeflow RAG Pipeline")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline.py")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Kubeflow host URL")
    parser.add_argument("--ui-host", default=None, help="Kubeflow UI host URL for browser links")
    parser.add_argument("--namespace", default=None, help="Kubeflow namespace for preflight checks")
    parser.add_argument("--auto-fix-namespace", action="store_true", help="Retry preflight once with detected namespace suggestion")
    parser.add_argument("--experiment", default="rag-experiment", help="KFP experiment name")
    parser.add_argument("--compile-only", action="store_true", help="Only compile, don't run")
    parser.add_argument("--strict-preflight", action="store_true", help="Fail with non-zero exit code when preflight checks fail")
    parser.add_argument("--output", default=None, help="Output YAML path for compiled pipeline")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_env_file(env_path)

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {args.config}")
    else:
        print(f"⚠️  Config not found at {args.config}, using defaults")

    # Resolve KFP host
    config_host = resolve_env_placeholders(config.get("kubeflow", {}).get("host"))
    config_ui_host = resolve_env_placeholders(config.get("kubeflow", {}).get("ui_host"))
    config_namespace = resolve_env_placeholders(config.get("kubeflow", {}).get("namespace"))
    host = args.host or config_host or os.environ.get("KUBEFLOW_HOST")
    ui_host = args.ui_host or config_ui_host or os.environ.get("KUBEFLOW_UI_HOST")
    namespace = args.namespace or os.environ.get("KUBEFLOW_NAMESPACE") or config_namespace or "kubeflow"
    if isinstance(host, str):
        host = host.strip()
    if isinstance(host, str) and ENV_PLACEHOLDER_PATTERN.search(host):
        print(f"⚠️  Kubeflow host contains unresolved placeholders: {host}")
        host = None

    # Load pipeline module
    print(f"🔧 Loading pipeline: {args.pipeline}")
    module = load_pipeline_module(args.pipeline)

    # Compile
    output = args.output or str(Path(args.pipeline).with_suffix(".yaml"))
    if hasattr(module, "compile_pipeline"):
        module.compile_pipeline(output)
        print(f"📦 Pipeline compiled → {output}")
    else:
        print("⚠️  No compile_pipeline function found in module")

    if args.compile_only:
        print("Compile-only mode. Done.")
        return

    # Run
    if not host:
        print("❌ No Kubeflow host specified. Use --host or set KUBEFLOW_HOST env variable")
        if args.strict_preflight:
            raise SystemExit(2)
        print("   Running compile-only...")
        return

    preflight_ok, suggested_namespace = preflight_kubeflow_host(host, namespace)
    if not preflight_ok and suggested_namespace and args.auto_fix_namespace:
        print(f"🔁 Auto-fix: retrying preflight with namespace '{suggested_namespace}'")
        namespace = suggested_namespace
        preflight_ok, suggested_namespace = preflight_kubeflow_host(host, namespace)

    if not preflight_ok:
        if suggested_namespace:
            retry_command = build_namespace_retry_command(suggested_namespace)
            print("   Suggested re-run command:")
            print(f"   {retry_command}")
        if args.strict_preflight:
            raise SystemExit(2)
        print("   Running compile-only...")
        return

    if hasattr(module, "run_pipeline"):
        print(f"🚀 Submitting pipeline to {host}...")
        run = module.run_pipeline(host, args.config, args.experiment)
        print(f"✅ Pipeline run submitted! Run ID: {run.run_id}")
        run_view_url, ui_message = resolve_run_view_url(host, run.run_id, ui_host)
        if run_view_url:
            print(f"   View at: {run_view_url}")
        if ui_message:
            print(ui_message)
    else:
        print("⚠️  No run_pipeline function found in module")


if __name__ == "__main__":
    main()
