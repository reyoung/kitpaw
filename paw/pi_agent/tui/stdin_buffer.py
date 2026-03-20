from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

ESC = "\x1b"
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"


def _is_complete_csi_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}["):
        return "complete"
    if len(data) < 3:
        return "incomplete"

    payload = data[2:]
    last_char = payload[-1]
    last_code = ord(last_char)
    if not 0x40 <= last_code <= 0x7E:
        return "incomplete"

    if payload.startswith("<"):
        if len(payload) == 1:
            return "incomplete"
        if payload[-1] in {"M", "m"}:
            parts = payload[1:-1].split(";")
            if len(parts) == 3 and all(part.isdigit() for part in parts):
                return "complete"
        return "incomplete"

    return "complete"


def _is_complete_osc_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}]"):
        return "complete"
    return "complete" if data.endswith(f"{ESC}\\") or data.endswith("\x07") else "incomplete"


def _is_complete_dcs_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}P"):
        return "complete"
    return "complete" if data.endswith(f"{ESC}\\") else "incomplete"


def _is_complete_apc_sequence(data: str) -> str:
    if not data.startswith(f"{ESC}_"):
        return "complete"
    return "complete" if data.endswith(f"{ESC}\\") else "incomplete"


def _is_complete_sequence(data: str) -> str:
    if not data.startswith(ESC):
        return "not-escape"
    if len(data) == 1:
        return "incomplete"

    after_esc = data[1:]
    if after_esc.startswith("["):
        if after_esc.startswith("[M"):
            return "complete" if len(data) >= 6 else "incomplete"
        return _is_complete_csi_sequence(data)
    if after_esc.startswith("]"):
        return _is_complete_osc_sequence(data)
    if after_esc.startswith("P"):
        return _is_complete_dcs_sequence(data)
    if after_esc.startswith("_"):
        return _is_complete_apc_sequence(data)
    if after_esc.startswith("O"):
        return "complete" if len(after_esc) >= 2 else "incomplete"
    if len(after_esc) == 1:
        return "complete"
    return "complete"


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    sequences: list[str] = []
    pos = 0

    while pos < len(buffer):
        remaining = buffer[pos:]
        if remaining.startswith(ESC):
            seq_end = 1
            while seq_end <= len(remaining):
                candidate = remaining[:seq_end]
                status = _is_complete_sequence(candidate)
                if status == "complete":
                    sequences.append(candidate)
                    pos += seq_end
                    break
                if status == "incomplete":
                    seq_end += 1
                    continue

                sequences.append(candidate)
                pos += seq_end
                break

            if seq_end > len(remaining):
                return sequences, remaining
        else:
            sequences.append(remaining[0])
            pos += 1

    return sequences, ""


class StdinBuffer:
    def __init__(self, timeout: float = 0.01) -> None:
        self._timeout = timeout
        self._buffer = ""
        self._timer: threading.Timer | None = None
        self._listeners: dict[str, list[Callable[..., None]]] = defaultdict(list)
        self._paste_mode = False
        self._paste_buffer = ""

    def on(self, event: str, listener: Callable[..., None]) -> None:
        self._listeners[event].append(listener)

    def _emit(self, event: str, *args: object) -> None:
        for listener in self._listeners.get(event, []):
            listener(*args)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def process(self, data: str | bytes) -> None:
        self._cancel_timer()

        if isinstance(data, bytes):
            if len(data) == 1 and data[0] > 127:
                string = f"\x1b{chr(data[0] - 128)}"
            else:
                string = data.decode()
        else:
            string = data

        if not string and not self._buffer:
            self._emit("data", "")
            return

        self._buffer += string

        if self._paste_mode:
            self._paste_buffer += self._buffer
            self._buffer = ""
            self._drain_paste_buffer()
            return

        start_index = self._buffer.find(BRACKETED_PASTE_START)
        if start_index != -1:
            if start_index > 0:
                sequences, _ = _extract_complete_sequences(self._buffer[:start_index])
                for sequence in sequences:
                    self._emit("data", sequence)

            self._buffer = self._buffer[start_index + len(BRACKETED_PASTE_START) :]
            self._paste_mode = True
            self._paste_buffer = self._buffer
            self._buffer = ""
            self._drain_paste_buffer()
            return

        sequences, remainder = _extract_complete_sequences(self._buffer)
        self._buffer = remainder
        for sequence in sequences:
            self._emit("data", sequence)

        if self._buffer:
            self._timer = threading.Timer(self._timeout, self._flush_and_emit)
            self._timer.daemon = True
            self._timer.start()

    def _drain_paste_buffer(self) -> None:
        end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
        if end_index == -1:
            return

        pasted_content = self._paste_buffer[:end_index]
        remaining = self._paste_buffer[end_index + len(BRACKETED_PASTE_END) :]
        self._paste_mode = False
        self._paste_buffer = ""
        self._emit("paste", pasted_content)
        if remaining:
            self.process(remaining)

    def _flush_and_emit(self) -> None:
        self._timer = None
        for sequence in self.flush():
            self._emit("data", sequence)

    def flush(self) -> list[str]:
        self._cancel_timer()
        if not self._buffer:
            return []
        sequences = [self._buffer]
        self._buffer = ""
        return sequences

    def clear(self) -> None:
        self._cancel_timer()
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""

    def get_buffer(self) -> str:
        return self._buffer

    def destroy(self) -> None:
        self.clear()
