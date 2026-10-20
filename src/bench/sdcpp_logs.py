"""Read the per-stage timings of an image generation from the sd-server docker logs."""
import os
import re
import subprocess

CONTAINER = os.getenv("IMAGE_CONTAINER", "test-project-image-1")

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


def read_logs(since_unix: float) -> str:
    """Return the container logs written since `since_unix` (empty string on failure)."""
    cmd = ["docker", "logs", "--since", str(int(since_unix)), CONTAINER]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout + result.stderr  # sd-server writes its logs to stderr


def sdcpp_stage_times(since_unix: float) -> dict[str, float]:
    """Parse stage timings of the run started at `since_unix`. Missing stages are left out."""
    logs = read_logs(since_unix)
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
