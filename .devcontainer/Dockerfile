FROM mcr.microsoft.com/devcontainers/miniconda:1-ubuntu-22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "fourier-mellin", "/bin/bash", "-lc"]
