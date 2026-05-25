#!/usr/bin/env bash
# RunPod orchestrator — start vLLM, run a benchmark sweep, run quality eval,
# stop the server, exit.
#
# Designed to run end-to-end with zero manual intervention. The Phase 6
# pre-flight rehearses this exact script on M1 against a tiny model via the
# --rehearsal flag before any GPU is rented.
#
# Usage:
#   ./deploy/runpod-run.sh --variant bf16         # full BF16 sweep + eval
#   ./deploy/runpod-run.sh --variant awq          # full AWQ-INT4 sweep + eval
#   ./deploy/runpod-run.sh --rehearsal            # M1 dress rehearsal, tiny model

set -euo pipefail

VARIANT=""
REHEARSAL=0
SKIP_EVAL=0
SERVER_READY_TIMEOUT=600     # seconds — full BF16 8B can take a few minutes
SERVER_PORT=8000

usage() {
    cat <<USAGE >&2
Usage: $0 [--variant bf16|awq | --rehearsal] [--skip-eval]
  --variant     Which model variant to benchmark + eval. Required unless --rehearsal.
  --rehearsal   Use tiny model + smoke configs. Validates the full pipeline.
  --skip-eval   Skip the quality eval step (e.g. if it ran in a prior run).
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"; shift 2 ;;
        --rehearsal)
            REHEARSAL=1; shift ;;
        --skip-eval)
            SKIP_EVAL=1; shift ;;
        -h|--help)
            usage ;;
        *)
            echo "[runpod-run] unknown arg: $1" >&2; usage ;;
    esac
done

if (( REHEARSAL == 0 )) && [[ -z "$VARIANT" ]]; then
    echo "[runpod-run] --variant is required (or pass --rehearsal)" >&2
    usage
fi

if (( REHEARSAL )); then
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    QUANTIZATION="${QUANTIZATION:-none}"
    BENCH_CONFIG="configs/bench-smoke.yaml"
    EVAL_CONFIG="configs/eval-smoke.yaml"
    LABEL="rehearsal"
elif [[ "$VARIANT" == "bf16" ]]; then
    MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    QUANTIZATION="none"
    BENCH_CONFIG="configs/bench-full.yaml"
    EVAL_CONFIG="configs/eval-full-bf16.yaml"
    LABEL="bf16"
elif [[ "$VARIANT" == "awq" ]]; then
    MODEL_ID="${MODEL_ID:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    QUANTIZATION="awq"
    BENCH_CONFIG="configs/bench-full-awq.yaml"
    EVAL_CONFIG="configs/eval-full-awq.yaml"
    LABEL="awq"
else
    echo "[runpod-run] unknown variant: $VARIANT (expected bf16 or awq)" >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server-$LABEL.log"

cleanup() {
    local code=$?
    echo "" >&2
    echo "[runpod-run] cleanup (exit code $code)" >&2
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[runpod-run] stopping vllm server (pid $SERVER_PID)" >&2
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        # Give vllm a chance to shut down cleanly; force after 10s.
        for _ in $(seq 1 10); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[runpod-run] force killing vllm" >&2
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    exit $code
}
trap cleanup EXIT INT TERM

echo "[runpod-run] variant=$LABEL"
echo "[runpod-run] model=$MODEL_ID"
echo "[runpod-run] quantization=$QUANTIZATION"
echo "[runpod-run] max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS"
echo "[runpod-run] bench=$BENCH_CONFIG eval=$EVAL_CONFIG"
echo "[runpod-run] server_log=$SERVER_LOG"
echo ""

# Start vLLM in the background.
echo "[runpod-run] starting vllm…"
MODEL_ID="$MODEL_ID" \
    QUANTIZATION="$QUANTIZATION" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    SERVING_PORT="$SERVER_PORT" \
    SERVING_HOST="0.0.0.0" \
    uv run python -m scripts.serve > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[runpod-run] vllm pid=$SERVER_PID"

# Wait for /health.
echo "[runpod-run] waiting for /health (timeout ${SERVER_READY_TIMEOUT}s)…"
SECONDS=0
until curl -sf "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; do
    if (( SECONDS > SERVER_READY_TIMEOUT )); then
        echo "[runpod-run] timed out waiting for vllm" >&2
        tail -50 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[runpod-run] vllm process died during startup" >&2
        tail -50 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done
echo "[runpod-run] /health is up after ${SECONDS}s"

# Benchmark.
echo ""
echo "[runpod-run] benchmark sweep…"
uv run python -m scripts.bench --config "$BENCH_CONFIG"

# Quality eval (skippable for fast iteration).
if (( SKIP_EVAL == 0 )); then
    echo ""
    echo "[runpod-run] quality eval…"
    uv run python -m scripts.eval --config "$EVAL_CONFIG"
fi

echo ""
echo "[runpod-run] done."
