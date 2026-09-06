FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv curl git \
    build-essential cmake nodejs npm zsh

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (optional but recommended)
RUN groupadd -g 1001 claude
RUN useradd -m -s /bin/zsh -u 1001 -g 1001 claude
ENV NPM_CONFIG_PREFIX=/home/claude/.npm-global
ENV PATH="/home/claude/.npm-global/bin:${PATH}"
RUN mkdir -p /home/claude/.npm-global && chown -R claude:claude /home/claude/.npm-global

USER claude
WORKDIR /workspace

# Set up Python venv
RUN python3 -m venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["/bin/zsh"]
