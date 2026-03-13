
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import uvicorn
from configs import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=None)
    parser.add_argument("--port",   default=None, type=int)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port

    print(f"\nStarting Image Captioning API")
    print(f"  API   → http://{host}:{port}")
    print(f"  Docs  → http://{host}:{port}/docs")
    print(f"  UI    → open frontend/index.html in browser\n")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()