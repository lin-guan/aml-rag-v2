from __future__ import annotations

import json
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=4) as response:
        body = json.load(response)
        if response.status // 100 != 2 or body.get("status") != "ok":
            raise RuntimeError("unhealthy response")
except Exception:
    sys.exit(1)
