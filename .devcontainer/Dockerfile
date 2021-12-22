FROM mcr.microsoft.com/vscode/devcontainers/universal:1.6.5-linux

USER root

# Install Tox
RUN pip install -U tox

RUN pip install pip-tools

# Install pre-commit
RUN pip install pre-commit

# Install Python 3.7
RUN sudo add-apt-repository ppa:deadsnakes/ppa \
    && sudo apt-get update \
    && sudo apt-get install -y python3.7

USER codespace
