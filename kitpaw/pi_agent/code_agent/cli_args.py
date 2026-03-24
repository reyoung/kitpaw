from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi")
    parser.add_argument("command", nargs="?")
    parser.add_argument("messages", nargs="*")
    parser.add_argument("--mode", choices=["text", "json", "rpc"], default="text")
    parser.add_argument("--agent", choices=["pi", "zed", "codex"], default="pi")
    parser.add_argument("--print", "-p", dest="print_mode", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--theme")
    parser.add_argument("--api-key")
    parser.add_argument("--thinking")
    parser.add_argument("--continue", "-c", dest="continue_session", action="store_true")
    parser.add_argument("--resume", "-r", nargs="?", const="__latest__")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--session-dir")
    parser.add_argument("--export")
    parser.add_argument("--tools")
    parser.add_argument("--local", "-l", action="store_true")
    return parser
