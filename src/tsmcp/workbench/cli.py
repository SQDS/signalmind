"""Start the local workbench with a single command."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--repo", type=Path, default=Path.cwd())
    serve.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(args.repo, args.state_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
