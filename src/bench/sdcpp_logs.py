"""Read the per-stage timings of an image generation from the sd-server logs (docker or k3s)."""
import os
import re
import subprocess
from datetime import UTC, datetime

# "docker" (compose) or "k8s" (k3s)
LOG_SOURCE = os.getenv("IMAGE_LOG_SOURCE", "docker")
CONTAINER = os.getenv("IMAGE_CONTAINER", "test-project-image-1")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "ai-models")
K8S_TARGET = os.getenv("IMAGE_K8S_TARGET", "deploy/image")
LOGS_TIMEOUT_S = 15

# metric name -> regex matching the sd-server log line (value in seconds)
STAGE_PATTERNS = {
    "encode_s": r"get_learned_condition completed, taking ([\d.]+)s",
    "sampling_s": r"sampling completed, taking ([\d.]+)s",
    "vae_decode_s": r"decode_first_stage completed, taking ([\d.]+)s",
    "server_total_s": r"generate_image completed in ([\d.]+)s",
}
# Weights can be loaded several times during one run, so these values are summed
WEIGHTS_LOAD_PATTERN = r"loading tensors completed, taking ([\d.]+)s"
# Same prompt as one of the last 4 requests -> text encoding is skipped
CACHE_HIT_LINE = "conditioning cache hit"


def docker_logs_cmd(since_unix: float) -> list[str]:
    """Build the docker logs command for the compose container."""
    return ["docker", "logs", "--since", str(int(since_unix)), CONTAINER]


def k8s_logs_cmd(since_unix: float) -> list[str]:
    """Build the kubectl logs command for the image deployment in k3s."""
    since = datetime.fromtimestamp(int(since_unix), tz=UTC)
    since_rfc3339 = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ["kubectl", "-n", K8S_NAMESPACE, "logs", K8S_TARGET, "--since-time", since_rfc3339]


def read_logs(since_unix: float) -> str:
    """Return the sd-server logs written since `since_unix` (empty string on failure)."""
    if LOG_SOURCE == "k8s":
        cmd = k8s_logs_cmd(since_unix)
    else:
        cmd = docker_logs_cmd(since_unix)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=LOGS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return ""
    # docker splits stdout/stderr (sd-server logs to stderr); kubectl merges both into stdout
    return result.stdout + result.stderr


def parse_stage_times(logs: str) -> dict[str, float]:
    """Parse stage timings from raw sd-server logs. Missing stages are left out."""
    times = {}

    for name, pattern in STAGE_PATTERNS.items():
        matches = re.findall(pattern, logs)
        if matches:
            times[name] = float(matches[-1])  # last match = most recent run

    loads = re.findall(WEIGHTS_LOAD_PATTERN, logs)
    if loads:
        times["weights_load_s"] = round(sum(float(x) for x in loads), 2)

    if "encode_s" in times:
        times["cache_hit"] = 1 if CACHE_HIT_LINE in logs else 0

    # The text encoder runs on CPU (--backend te=cpu)
    if "encode_s" in times and times.get("server_total_s"):
        times["cpu_share_pct"] = round(100 * times["encode_s"] / times["server_total_s"], 1)
    return times


def sdcpp_stage_times(since_unix: float) -> dict[str, float]:
    """Stage timings of the run started at `since_unix`."""
    return parse_stage_times(read_logs(since_unix))
