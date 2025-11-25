ARG VARIANT=21-bookworm
FROM mcr.microsoft.com/devcontainers/java:1-${VARIANT}

ARG INSTALL_MAVEN="false"
ARG MAVEN_VERSION=""

ARG INSTALL_GRADLE="false"
ARG GRADLE_VERSION=""

RUN if [ "${INSTALL_MAVEN}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/sdkman/bin/sdkman-init.sh && sdk install maven \"${MAVEN_VERSION}\""; fi \
    && if [ "${INSTALL_GRADLE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/sdkman/bin/sdkman-init.sh && sdk install gradle \"${GRADLE_VERSION}\""; fi

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Install required packages
RUN apt-get update \
    && apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release sudo

# Install additional dependencies required by Apache Tika (document parser for Ingestor)
RUN apt-get update \
    && apt-get install -y tesseract-ocr libxtst6 libxi6 git python3-venv

# Install Vulkan libraries (required for ONNX Runtime GPU support)
# RUN apt-get install -y vulkan-utils libvulkan1

# Install NVIDIA CUDA 12 libraries for GPU acceleration
# This installs the runtime libraries required by ONNX Runtime GPU
# RUN apt-get update \
#     && apt-get install -y wget \
#     && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
#     && dpkg -i cuda-keyring_1.1-1_all.deb \
#     && rm cuda-keyring_1.1-1_all.deb \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#         cuda-cudart-12-6 \
#         libcublas-12-6 \
#         libcublaslt-12-6 \
#         libcufft-12-6 \
#         libcurand-12-6 \
#         libcusparse-12-6 \
#         libcudnn9-cuda-12 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Set up environment for CUDA
# ENV PATH=/usr/local/cuda-12/bin${PATH:+:${PATH}}
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

ARG USERNAME=vscode

# Add the user to the sudo group and configure NOPASSWD for sudo
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

