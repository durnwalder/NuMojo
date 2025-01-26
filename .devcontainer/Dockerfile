FROM ghcr.io/modular/magic:latest

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential gdb cmake cppcheck git zip unzip

ENV MODULAR_HOME="/home/codespace/.modular"
ENV PATH="${MODULAR_HOME}/bin:${MODULAR_HOME}/pkg/packages.modular.com_mojo/bin:${PATH}"

USER vscode
WORKDIR /home/vscode