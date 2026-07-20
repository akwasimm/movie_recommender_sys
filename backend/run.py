import os
import uvicorn
from src.config import get_config

cfg = get_config()
port = int(os.environ.get("PORT", cfg["api"]["port"]))
reload = os.environ.get("RELOAD", "0") == "1"

if __name__ == "__main__":
    print(f"\n  CineScope v1.0.0")
    print(f"  http://localhost:{port}/app\n")
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=reload)
