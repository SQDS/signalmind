"""Self-describing, process-independent evidence refs: `<id>:<kind>@<where>`."""


def ref(sid: str, kind: str, where: str) -> str:
    return f"{sid}:{kind}@{where}"
