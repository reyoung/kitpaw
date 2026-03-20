"""Lightweight ``paw.pi_agent`` package marker.

Keep the root package free of eager submodule imports so platform-specific
dependencies in optional surfaces (for example the TUI) do not affect users
who only import other subpackages such as ``paw.pi_agent.ai``.
"""

from __future__ import annotations
