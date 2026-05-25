# Monitoring

vLLM exports Prometheus metrics on the same port as the OpenAI API at `/metrics`. The local docker-compose stack runs:

- **Prometheus** (`:9090`) — scrapes the vLLM CPU container every 5s.
- **Grafana** (`:3000`) — auto-provisions the Forge dashboard and the Prometheus datasource at startup.

## Files

| Path | Purpose |
|---|---|
| `prometheus.yml` | Scrape config — points Prometheus at the vLLM container. |
| `grafana/dashboards/forge.json` | The Forge dashboard. 9 panels (in-flight requests, KV cache, TTFT / TPOT / e2e p50/p95/p99, throughput, success rate, preemptions, total tokens). |
| `grafana/provisioning/dashboards/forge.yaml` | Tells Grafana where to find the dashboard JSONs. |
| `grafana/provisioning/datasources/prometheus.yaml` | Wires the Prometheus datasource. |

## Running locally on M1

```bash
docker compose up -d
# Wait for vllm-cpu's healthcheck to go green (~2 min on M1).
open http://localhost:3000  # admin/admin — anonymous viewer also enabled
```

Then send traffic at the vLLM container (`curl http://localhost:8000/v1/chat/completions ...`) and the dashboard will populate.

## On RunPod

The production CUDA `Dockerfile` doesn't bundle Prometheus + Grafana — they run on the host. For the paid run, either:

1. Forward the pod's `:8000/metrics` to a local Prometheus on your laptop.
2. Or use RunPod's built-in monitoring (CPU/GPU graphs) plus the per-run benchmark JSONs.

The screenshots in the README come from a brief on-pod stress run with this dashboard pulled up. Methodology recorded in `docs/methodology.md`.
