# syntax=docker/dockerfile:1
#
# The web workbench -- the page with its WebAssembly engine, plus the local files server that
# lists and exports this machine's checkpoints -- built from the GitHub repository.
#
#     docker build -t deepstruggle-web deploy/web
#     docker run -d -p 8000:8000 -v /path/to/checkpoints:/data/checkpoints:ro \
#                -v deepstruggle-replays:/data/replays deepstruggle-web
#
# The source is not taken from the build context: it is fetched from REPO at REF by BuildKit's git
# source (`ADD <repo>#<ref>`), which resolves the ref to a commit on every build -- so a new push
# to the branch invalidates the cache, and an unchanged branch reuses it. The build context is
# only this directory, which keeps a multi-gigabyte `data/` tree from being shipped to the daemon.
#
#     --build-arg REF=<branch|tag|commit>      what to deploy (default: main)
#     --build-arg REPO=<git url>               a fork or mirror
#     --build-arg ENGINE_MARCH=<gcc -march>    see the engine stage (default: portable)
#
# Checkpoints are not in the image. Mount a checkpoints tree at /data/checkpoints -- the same
# layout as data/checkpoints, `<run>/snapshot_*.pt` -- and the analysis panel lists it; the server
# exports a checkpoint to ONNX (cached in /data/onnx_cache) the first time the page asks for it.
# Replays in /data/replays are listed for watching. See deploy/web/README.md.

ARG REPO=https://github.com/mihaild/DeepStruggle.git
ARG REF=main
ARG PYTHON_IMAGE=python:3.14-slim
ARG NODE_IMAGE=node:22-slim
# Pinned like tools/scripts/install_emsdk.sh: the browser engine is checked bit for bit against
# the native one with this compiler (tests/web/test_wasm_engine.py).
ARG EMSDK_IMAGE=emscripten/emsdk:6.0.10

# -- source ---------------------------------------------------------------------------------------
FROM scratch AS source
ARG REPO
ARG REF
# No submodules: external/struggler is the differential-testing reference engine, not a runtime
# dependency of anything the server imports.
ADD --keep-git-dir=false ${REPO}#${REF} /

# -- the engine for the page (WebAssembly) -------------------------------------------------------
FROM ${EMSDK_IMAGE} AS wasm
WORKDIR /src
COPY --from=source / ./
# The fingerprint of these sources is baked in, so the page can tell it runs this build.
RUN emcmake cmake -B build/wasm -S . -DCMAKE_BUILD_TYPE=Release \
        -DTS_ENGINE_FINGERPRINT="$(PYTHONPATH=/src python3 tools/lib/engine_fingerprint.py)" \
 && cmake --build build/wasm --target ts_engine_wasm -j"$(nproc)"

# -- web UI ---------------------------------------------------------------------------------------
FROM ${NODE_IMAGE} AS ui
WORKDIR /src/web/ui
COPY --from=source /web/ui/package.json /web/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY --from=source /web/ui/ ./
# The page bundles rules/map.json and rules/cards.json from outside web/ui.
COPY --from=source /rules /src/rules
COPY --from=wasm /src/web/ui/public/engine ./public/engine
RUN npm run build

# -- ts_engine ------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS engine
# The repository's CMake files compile with -march=native, which is right for a training box and
# wrong for an image: built on one CPU, it dies with SIGILL on an older one. The flag is swapped
# here, in the image's copy only, for a portable baseline (x86-64-v2: any x86 server since ~2009;
# armv8-a on ARM). Pass ENGINE_MARCH=native when the image is built on the machine it runs on.
ARG ENGINE_MARCH=
# clang: the engine is built with clang, and CMake refuses any other compiler. build-essential
# for make, libstdc++ and libgomp, which the batch runner links (it shares torch's OpenMP pool).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential clang cmake \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# nanobind is a build-time dependency only; CMake finds it through the interpreter.
RUN pip install --no-cache-dir "nanobind>=2.15.0"
WORKDIR /src
COPY --from=source / ./
RUN march="${ENGINE_MARCH}"; \
    if [ -z "$march" ]; then \
      case "$(uname -m)" in \
        x86_64) march=x86-64-v2 ;; \
        aarch64|arm64) march=armv8-a ;; \
        *) march=native ;; \
      esac; \
    fi; \
    echo "ts_engine: -march=$march"; \
    sed -i "s/-march=native/-march=$march/" CMakeLists.txt engine/CMakeLists.txt \
 && cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/venv/bin/python \
 && cmake --build build/release --target ts_engine -j"$(nproc)" \
 && mkdir /out && cp build/release/ts_engine*.so /out/

# -- runtime --------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
# libgomp: the batch runner calls GCC's OpenMP runtime directly, under any compiler.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Only what the server imports: fastapi/uvicorn, and for exporting a checkpoint to ONNX torch
# (CPU wheel: an export needs no GPU, and the CUDA wheels would add gigabytes), onnx/onnxscript,
# and onnxruntime, which verifies every export against torch before it is served.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.13.0" \
 && pip install --no-cache-dir "numpy>=2.5.0" "fastapi>=0.141.0" "uvicorn[standard]>=0.52.0" \
        "pydantic>=2.13.0" "onnx>=1.23.0" "onnxscript>=0.7.2" "onnxruntime>=1.30.0"

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data/checkpoints /data/replays /data/onnx_cache \
 && chown -R app:app /data
WORKDIR /app
COPY --from=source / ./
COPY --from=engine /out/ ./build/release/
COPY --from=ui /src/web/ui/dist ./web/ui/dist

ENV PYTHONPATH=/app:/app/build/release \
    PYTHONUNBUFFERED=1 \
    TS_DATA_ROOT=/data \
    TS_CHECKPOINTS_DIR=/data/checkpoints \
    TS_REPLAYS_DIR=/data/replays \
    TS_ONNX_CACHE_DIR=/data/onnx_cache

USER app
VOLUME ["/data/replays", "/data/onnx_cache"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/local/info', timeout=4)"]
CMD ["python", "-m", "web.server.main", "--host", "0.0.0.0", "--port", "8000", "--log-level", "INFO"]
