FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends  ca-certificates python3-setuptools python3-wheel python3-pip build-essential libmariadb-dev unattended-upgrades && \
    unattended-upgrade && \
    python3 -m pip install -r requirements.txt