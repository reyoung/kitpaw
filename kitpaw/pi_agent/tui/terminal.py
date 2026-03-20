from __future__ import annotations

import asyncio
import os
import signal
import sys
import termios
import threading
import time
import tty
from typing import Callable, Protocol

from .keys import set_kitty_protocol_active
from .stdin_buffer import StdinBuffer


class Terminal(Protocol):
    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...
    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None: ...
    def write(self, data: str) -> None: ...
    @property
    def columns(self) -> int: ...
    @property
    def rows(self) -> int: ...
    @property
    def kitty_protocol_active(self) -> bool: ...
    def move_by(self, lines: int) -> None: ...
    def hide_cursor(self) -> None: ...
    def show_cursor(self) -> None: ...
    def clear_line(self) -> None: ...
    def clear_from_cursor(self) -> None: ...
    def clear_screen(self) -> None: ...
    def set_title(self, title: str) -> None: ...


class ProcessTerminal:
    def __init__(self) -> None:
        self._was_raw = False
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active = False
        self._modify_other_keys_active = False
        self._stdin_buffer: StdinBuffer | None = None
        self._stdin_data_handler: Callable[[str], None] | None = None
        self._kitty_query_timer: threading.Timer | None = None
        self._kitty_query_timeout_s = 0.15
        self._write_log_path = os.getenv("PI_TUI_WRITE_LOG", "")

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    @property
    def columns(self) -> int:
        return int(getattr(sys.stdout, "columns", 80) or 80)

    @property
    def rows(self) -> int:
        return int(getattr(sys.stdout, "rows", 24) or 24)

    def _set_raw_mode(self, enabled: bool) -> None:
        stdin = sys.stdin
        if hasattr(stdin, "setraw"):
            stdin.setraw(enabled)
            return

        if hasattr(stdin, "setRawMode"):
            stdin.setRawMode(enabled)
            return

        fd = stdin.fileno()
        if enabled:
            tty.setraw(fd)
        else:
            try:
                attrs = termios.tcgetattr(fd)
            except Exception:
                return
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize
        self._was_raw = bool(getattr(sys.stdin, "is_raw", getattr(sys.stdin, "isRaw", False)))

        try:
            termios.tcgetattr(sys.stdin.fileno())
            self._set_raw_mode(True)
        except Exception:
            self._was_raw = False

        if hasattr(sys.stdin, "set_encoding"):
            sys.stdin.set_encoding("utf8")
        elif hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")

        if hasattr(sys.stdin, "resume"):
            sys.stdin.resume()

        self.write("\x1b[?2004h")

        if hasattr(sys.stdout, "on"):
            sys.stdout.on("resize", on_resize)

        if os.name != "nt":
            try:
                os.kill(os.getpid(), signal.SIGWINCH)
            except Exception:
                pass

        self._query_and_enable_kitty_protocol()

    def _setup_stdin_buffer(self) -> None:
        self._stdin_buffer = StdinBuffer(timeout=0.01)
        kitty_response_pattern = "\x1b[?"

        def on_data(sequence: str) -> None:
            if not self._kitty_protocol_active and sequence.startswith(kitty_response_pattern) and sequence.endswith("u"):
                self._kitty_protocol_active = True
                set_kitty_protocol_active(True)
                self._cancel_kitty_query_timer()
                self.write("\x1b[>7u")
                return
            if self._input_handler is not None:
                self._input_handler(sequence)

        def on_paste(content: str) -> None:
            if self._input_handler is not None:
                self._input_handler(f"\x1b[200~{content}\x1b[201~")

        self._stdin_buffer.on("data", on_data)
        self._stdin_buffer.on("paste", on_paste)
        self._stdin_data_handler = lambda data: self._stdin_buffer.process(data)

    def _cancel_kitty_query_timer(self) -> None:
        if self._kitty_query_timer is not None:
            self._kitty_query_timer.cancel()
            self._kitty_query_timer = None

    def _enable_modify_other_keys_fallback(self) -> None:
        if self._kitty_protocol_active or self._modify_other_keys_active:
            return
        self.write("\x1b[>4;2m")
        self._modify_other_keys_active = True

    def _query_and_enable_kitty_protocol(self) -> None:
        self._setup_stdin_buffer()
        if self._stdin_data_handler is not None and hasattr(sys.stdin, "on"):
            sys.stdin.on("data", self._stdin_data_handler)
        self.write("\x1b[?u")
        self._kitty_query_timer = threading.Timer(self._kitty_query_timeout_s, self._enable_modify_other_keys_fallback)
        self._kitty_query_timer.daemon = True
        self._kitty_query_timer.start()

    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
        if self._kitty_protocol_active:
            self.write("\x1b[<u")
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        if self._modify_other_keys_active:
            self.write("\x1b[>4;0m")
            self._modify_other_keys_active = False

        previous_handler = self._input_handler
        self._input_handler = None
        last_data_time = time.monotonic()

        def on_data(_data: str) -> None:
            nonlocal last_data_time
            last_data_time = time.monotonic()

        if hasattr(sys.stdin, "on"):
            sys.stdin.on("data", on_data)

        end_time = time.monotonic() + max_ms / 1000
        try:
            while True:
                now = time.monotonic()
                if now >= end_time:
                    break
                if now - last_data_time >= idle_ms / 1000:
                    break
                await asyncio.sleep(min(idle_ms / 1000, end_time - now))
        finally:
            if hasattr(sys.stdin, "remove_listener"):
                sys.stdin.remove_listener("data", on_data)
            self._input_handler = previous_handler

    def stop(self) -> None:
        self._cancel_kitty_query_timer()
        self.write("\x1b[?2004l")

        if self._kitty_protocol_active:
            self.write("\x1b[<u")
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        if self._modify_other_keys_active:
            self.write("\x1b[>4;0m")
            self._modify_other_keys_active = False

        if self._stdin_buffer is not None:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None

        if self._stdin_data_handler is not None and hasattr(sys.stdin, "remove_listener"):
            sys.stdin.remove_listener("data", self._stdin_data_handler)
            self._stdin_data_handler = None

        self._input_handler = None
        if self._resize_handler is not None and hasattr(sys.stdout, "remove_listener"):
            sys.stdout.remove_listener("resize", self._resize_handler)
            self._resize_handler = None

        if hasattr(sys.stdin, "pause"):
            sys.stdin.pause()

        try:
            self._set_raw_mode(self._was_raw)
        except Exception:
            pass

    def write(self, data: str) -> None:
        sys.stdout.write(data)
        if self._write_log_path:
            try:
                with open(self._write_log_path, "a", encoding="utf-8") as fh:
                    fh.write(data)
            except Exception:
                pass
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self.write(f"\x1b[{lines}B")
        elif lines < 0:
            self.write(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self.write(f"\x1b]0;{title}\x07")
