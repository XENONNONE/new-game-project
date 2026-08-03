# AGENTS.md — Development Workflow for GestureLSM Avatar Runtime

## Project Layout

```
new-game-project/
├── GestureLSM/              # PyTorch gesture generation model + inference runtime
│   ├── inference_runtime/   # Production HTTP server, pipeline, TTS, LLM, config
│   │   ├── config.py        # YAML + env-var configuration loading
│   │   ├── logging_config.py # Structured JSON/human logging with rotation
│   │   ├── rate_limit.py     # Rate limiter + circuit breaker utilities
│   │   ├── server.py         # HTTP server with SSE streaming, metrics, health
│   │   ├── pipeline.py       # GesturePipeline: MeanFlow + RVQ decoder + streaming
│   │   ├── retarget.py       # SMPL-X → VRM bone mapping + quaternion math
│   │   ├── audio.py          # WAV preprocessing, onset detection, windowing
│   │   ├── tts.py            # Kokoro/KittenTTS/command TTS backends
│   │   ├── conversation.py   # Speech plan contract (AvatarSpeechPlan)
│   │   ├── llm.py            # Qwen chat client
│   │   └── tests_contract.py # Unit tests for speech plan contract
│   ├── models/              # Model implementations (MeanFlow, LSM, Diffusion, VQ)
│   ├── configs_new/         # YAML model configs
│   ├── tests/               # Unit + integration tests
│   ├── mean_std/            # Normalization statistics
│   ├── requirements.txt
│   └── config.yaml.example  # All available configuration options
├── godot/                   # Godot 4.7 client (main.gd, avatar rendering)
├── scripts/                 # Launch, install, and service scripts
│   ├── start_avatar_server.sh
│   ├── install.sh
│   ├── avatar-server.service
│   └── smoke_avatar_server.py
├── models/                  # Downloaded model weights (LLM, TTS)
├── ckpt/                    # GestureLSM checkpoints
├── Dockerfile               # Container build
├── docker-compose.yml       # Avatar + Qwen services
├── .dockerignore            # Docker build exclusions
├── .env.example             # Environment variable reference
├── pyproject.toml           # Ruff, mypy, and pytest configuration
├── requirements-dev.txt     # Dev and testing dependencies
├── test.wav                 # Bundled test WAV for /infer_test
└── .github/workflows/ci.yml # CI: lint, test, type-check, security, docker
```

## Quick Start (Development)

```bash
# 1. Install Python dependencies (Python 3.12 recommended)
pip install -r GestureLSM/requirements.txt
pip install kittentts onnxruntime soundfile "misaki[en]"

# 2. Download models (see ANDROID_OFFLINE.md for details)
#    - GestureLSM checkpoints → ckpt/
#    - Qwen LLM → models/llm/
#    - KittenTTS → auto-download from HuggingFace on first use (no manual download needed)
#    - ONNX Runtime → pip install onnxruntime (optional, accelerates MeanFlow denoiser ~11x)
#    - ONNX model → run GestureLSM/tests/test_onnx_export.py to generate (models/meanflow_denoiser.int8.onnx)
#    - ONNX RVQ decoders → run GestureLSM/tests/test_rvq_onnx.py to generate (models/rvq_*_onnx)
#    - Enable in .env → AVATAR_PIPELINE_USE_ONNX=1
#    - Use 2-4 threads for ONNX inference (AVATAR_THREADS=4)

# 3. Run the server
sh scripts/start_avatar_server.sh

# 4. Run tests
python -m pytest GestureLSM/tests/ -v

# 5. Run the Godot client (Godot 4.7 editor)
#    Open project.godot in Godot, press Play

# Streaming endpoints:
#   POST /infer        - Full WAV → complete motion JSON
#   POST /infer_stream - Full WAV → SSE stream of gesture frames (real-time)
#   POST /infer_test   - Run inference on the bundled test.wav
#   POST /speak        - Speech plan → TTS → gesture motion
#   POST /chat         - Qwen reply → TTS → gesture motion
```

## Testing

```bash
# Run all tests
python -m pytest -v

# Run specific test module
python -m pytest GestureLSM/tests/test_conversation.py -v

# Run with coverage
pip install pytest-cov
python -m pytest --cov=inference_runtime --cov-report=term-missing

# Run only fast tests (skip integration)
python -m pytest -v -k "not server"
```

## Linting & Type Checking

```bash
# Install dev tools
pip install ruff mypy

# Lint
ruff check GestureLSM/

# Format
ruff format GestureLSM/

# Type check (requires torch stubs)
mypy GestureLSM/inference_runtime/ --ignore-missing-imports
```

## Configuration

The runtime reads configuration from (in priority order):
1. Environment variables (`AVATAR_SERVER_PORT`, `AVATAR_LLM_TIMEOUT`, etc.)
2. `config.yaml` in the project root or `GestureLSM/config.yaml`
3. Built-in defaults

See `GestureLSM/config.yaml.example` for all available options.

### Streaming Configuration

The `streaming` section controls the rolling/streaming gesture inference pipeline:

| Key | Env Var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `AVATAR_STREAMING_ENABLED` | `true` | Enable streaming inference |
| `window_samples` | `AVATAR_STREAMING_WINDOW_SAMPLES` | `68224` | Audio window size in samples (4.264s @ 16kHz) |
| `overlap_frames` | `AVATAR_STREAMING_OVERLAP_FRAMES` | `16` | Frame overlap between windows for blending |
| `stream_fps` | `AVATAR_STREAMING_STREAM_FPS` | `30` | Output motion FPS |
| `ladder_step` | `AVATAR_STREAMING_LADDER_STEP` | `1` | RDLA ladder step (1=no acceleration, 2=2× speedup) |
| `ladder_strategy` | `AVATAR_STREAMING_LADDER_STRATEGY` | `down` | Ladder strategy: `up` or `down` |
| `max_concurrent_windows` | `AVATAR_STREAMING_MAX_CONCURRENT_WINDOWS` | `1` | Parallel window processing |
| `chunk_duration_s` | `AVATAR_STREAMING_CHUNK_DURATION_S` | `4.264` | Audio chunk duration |
| `request_timeout_s` | `AVATAR_STREAMING_REQUEST_TIMEOUT_S` | `300.0` | Streaming request timeout |

## Logging

- Default: stderr, human-readable, INFO level
- JSON format: set `AVATAR_LOG_JSON=1`
- File output: set `AVATAR_LOG_FILE=/path/to/log`
- Log level: set `AVATAR_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`

## CI/CD

GitHub Actions runs on every push:
- `test` — pytest with coverage
- `lint` — ruff check and format
- `type-check` — mypy

## Deployment

### Docker
```bash
docker-compose up -d
```

### Systemd (Linux)
```bash
INSTALL_SYSTEMD=1 sh scripts/install.sh
systemctl start avatar-server
```

### Manual
```bash
sh scripts/start_avatar_server.sh
```

## Code Style

- Python: 4-space indent, type hints, docstrings
- No comments unless explaining non-obvious logic
- Use `logging.getLogger("avatar.<module>")` for all logging
- Follow the existing patterns in `inference_runtime/`
