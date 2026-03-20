from __future__ import annotations

import asyncio
import json
import sys


class RpcClient:
    def __init__(self, cli_path: str, cwd: str, env: dict[str, str] | None = None) -> None:
        self.cli_path = cli_path
        self.cwd = cwd
        self.env = env
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            self.cli_path,
            "--mode",
            "rpc",
            cwd=self.cwd,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        if self.process is None:
            return
        self.process.kill()
        await self.process.wait()

    async def send(self, payload: dict[str, object]) -> list[dict[str, object]]:
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        await self.process.stdin.drain()
        events: list[dict[str, object]] = []
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            event = json.loads(line)
            events.append(event)
            if event.get("type") == "response":
                return events
        return events
