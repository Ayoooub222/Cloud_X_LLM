"""Load the app configuration (conf/app.conf) and the vLLM API key (.env)."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

CONF_PATH = Path("conf/app.conf")
API_KEY_VAR = "VLLM_API_KEY"


class Loader:
    def load_conf(self, path: Path = CONF_PATH) -> dict:
        """Read the JSON configuration file."""
        return json.loads(path.read_text(encoding="utf-8"))

    def load_env(self) -> str | None:
        """Load .env if present (no error otherwise) and return the vLLM API key."""
        load_dotenv()
        return os.getenv(API_KEY_VAR)
