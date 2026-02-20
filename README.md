# vLLM Jina Embeddings Benchmark

Benchmark and smoke-test suite for [Jina Embeddings v5 Nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) served by [vLLM](https://docs.vllm.ai/).

## Prerequisites

- Docker with Compose v2
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.12+

## Quick Start

```bash
# 1. Build and start the vLLM server
make up-nvidia

# 2. Set up the Python test environment
make setup

# 3. (Optional) Pre-generate the test corpus — auto-generated on first run if missing
make generate

# 4. Run the tests
make test
```

## Makefile Targets

| Target       | Description                                            |
|--------------|--------------------------------------------------------|
| `build`      | Build the vLLM Docker image                            |
| `up-nvidia`  | Build and start the vLLM server (NVIDIA GPU)           |
| `down`       | Stop the vLLM server                                   |
| `logs`       | Tail the vLLM server logs                              |
| `pull`       | Pull latest base images                                |
| `setup`      | Create `.venv/` and install Python dependencies        |
| `generate`   | Pre-generate `test_texts.txt` (100k lines, ~207 MB)   |
| `test`       | Run the full test suite against the running server     |

## Test Suite

The test script (`test_embeddings.py`) runs four tests against the OpenAI-compatible `/v1/embeddings` endpoint on `localhost:8333`:

| Test | Description |
|------|-------------|
| 1 | Basic single embedding — verifies 768-dimension output |
| 2 | Matryoshka truncation — requests 128-dimension output |
| 3 | Batch input — sends 3 strings in one request |
| 4 | Async batch throughput — 1,000 batches of 100 texts with 4 concurrent async workers |

Test 4 reads from a pre-generated corpus (`test_texts.txt`) so results are reproducible across runs. The file is created automatically on the first run or explicitly with `make generate`.

A statistics summary is printed at the end:

```
============================================================
  TEST STATISTICS
============================================================
  Model:              jinaai/jina-embeddings-v5-text-nano-retrieval
  Endpoint:           http://0.0.0.0:8333/v1/embeddings
  Async workers:      4
  Batch size:         100
  Number of batches:  1000
  Total texts:        100,000
  Texts file:         /path/to/test_texts.txt
  Total wall time:    123.4s
  Batch test time:    120.1s
  Embeddings/sec:     832.6
  Tests passed:       4/4
============================================================
```

## Configuration

Edit the constants at the top of `test_embeddings.py`:

| Constant      | Default | Description                        |
|---------------|---------|------------------------------------|
| `BASE_URL`    | `http://0.0.0.0:8333/v1/embeddings` | vLLM endpoint |
| `NUM_WORKERS` | `4`     | Concurrent async workers for test 4 |
| `NUM_BATCHES` | `1000`  | Number of batches in test 4        |
| `BATCH_SIZE`  | `100`   | Texts per batch in test 4          |

## Docker

The `Dockerfile` extends `vllm/vllm-openai:latest` with updated `transformers` and `peft` packages required by the Jina v5 model. The Compose file mounts your local HuggingFace cache to avoid re-downloading model weights:

```bash
# Use a custom cache path
HF_CACHE=/path/to/cache make up-nvidia
```

## Project Structure

```
.
├── Dockerfile              # vLLM image with Jina v5 dependencies
├── Makefile                # Build, run, and test targets
├── README.md
├── compose.yml             # Docker Compose (nvidia profile)
├── requirements.txt        # Python dependencies (aiohttp, requests)
├── test_embeddings.py      # Test + benchmark script
└── test_texts.txt          # Generated test corpus (git-ignored)
```
