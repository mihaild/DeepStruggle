# Image for running training on rented GPU hosts.
#
# Built once and pushed to a registry, then referenced when launching an instance. The point is
# to pay for the PyTorch download (~2.5GB) and the apt toolchain once rather than on every
# instance: a run is only ~1.8h, so a ten-minute setup is 9% of it, and at six instances that is
# an hour of rental burned on `pip install`.
#
# The versions are pinned because the local stack is specific: Python 3.14, torch 2.13+cu130,
# Ubuntu 26.04. The nanobind extension links against the exact Python it was built for, so a
# mismatched base image produces an engine that will not import.
FROM nvidia/cuda:13.0.0-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.14 python3.14-dev python3.14-venv \
        build-essential clang cmake git rsync openssh-client ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ts

# Dependencies as their own layer, so a source change does not invalidate the torch download.
COPY requirements.txt /opt/ts/requirements.txt
RUN python3.14 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /opt/ts/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

# The source is NOT baked in: it is rsynced at launch so an image rebuild is not needed for
# every experiment. The engine is built on the instance by remote_bootstrap.sh, which is cheap
# (the C++ core links no CUDA) and guarantees the extension matches this image's Python.
CMD ["/bin/bash"]
