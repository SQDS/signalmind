"""TS-MCP: let LLMs analyze long time-series signals via handles, views and exact measurements."""

__version__ = "0.1.0"


def main() -> None:
    from tsmcp.server.app import main as _main

    _main()
