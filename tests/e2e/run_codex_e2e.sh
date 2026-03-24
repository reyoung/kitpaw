#!/usr/bin/env bash
#
# End-to-end comparison: Codex (Rust) vs kitpaw (Python) --agent codex
#
# Starts a recording proxy, runs the same prompt through both agents,
# then compares the captured requests.
#
# Usage:
#   ./tests/e2e/run_codex_e2e.sh "Explain what this repo does"
#
# Prerequisites:
#   - .env.local with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
#   - Codex binary built: ref/codex/codex-rs/target/release/codex-exec
#   - kitpaw installed in .venv

set -euo pipefail
cd "$(dirname "$0")/../.."

PROMPT="${1:-Reply with exactly the word pong.}"
PROXY_PORT_CODEX=19901
PROXY_PORT_KITPAW=19902
CODEX_OUTPUT="tests/e2e/codex_requests.jsonl"
KITPAW_OUTPUT="tests/e2e/kitpaw_requests.jsonl"

# Load env
if [ -f .env.local ]; then
    set -a; source .env.local; set +a
fi

UPSTREAM="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
echo "Upstream: $UPSTREAM"
echo "Model:    ${OPENAI_MODEL:-gpt-4o-mini}"
echo "Prompt:   $PROMPT"
echo ""

# Clean old captures
rm -f "$CODEX_OUTPUT" "$KITPAW_OUTPUT"

# --- Start proxy for Codex ---
echo "Starting Codex proxy on :$PROXY_PORT_CODEX..."
.venv/bin/python tests/e2e/recording_proxy.py \
    --upstream "$UPSTREAM" \
    --port "$PROXY_PORT_CODEX" \
    --output "$CODEX_OUTPUT" &
PROXY_PID_CODEX=$!
sleep 1

# --- Run Codex (Rust) ---
CODEX_BIN="ref/codex/codex-rs/target/release/codex-exec"
if [ -f "$CODEX_BIN" ]; then
    echo "Running Codex (Rust)..."
    OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT_CODEX" \
        timeout 60 "$CODEX_BIN" --prompt "$PROMPT" 2>/dev/null || true
    echo "Codex done."
else
    echo "WARNING: Codex binary not found at $CODEX_BIN"
    echo "  Build with: cd ref/codex/codex-rs && cargo build --release -p codex-exec"
    echo "  Skipping Codex capture."
fi

kill $PROXY_PID_CODEX 2>/dev/null || true
wait $PROXY_PID_CODEX 2>/dev/null || true
sleep 1

# --- Start proxy for kitpaw ---
echo ""
echo "Starting kitpaw proxy on :$PROXY_PORT_KITPAW..."
.venv/bin/python tests/e2e/recording_proxy.py \
    --upstream "$UPSTREAM" \
    --port "$PROXY_PORT_KITPAW" \
    --output "$KITPAW_OUTPUT" &
PROXY_PID_KITPAW=$!
sleep 1

# --- Run kitpaw ---
echo "Running kitpaw --agent codex..."
OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT_KITPAW" \
    timeout 60 .venv/bin/python -m kitpaw.pi_agent.code_agent \
    --agent codex -p "$PROMPT" 2>/dev/null || true
echo "kitpaw done."

kill $PROXY_PID_KITPAW 2>/dev/null || true
wait $PROXY_PID_KITPAW 2>/dev/null || true

# --- Compare ---
echo ""
echo "=========================================="
echo "COMPARISON"
echo "=========================================="
.venv/bin/python tests/e2e/compare_requests.py "$CODEX_OUTPUT" "$KITPAW_OUTPUT"
