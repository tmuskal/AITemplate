FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Install dependencies

WORKDIR /app
# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
RUN  apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    wget \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive https://github.com/tmuskal/AITemplate

WORKDIR /app/AITemplate/python

RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl --force-reinstall

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install diffusers transformers click einops


WORKDIR /app/AITemplate
ENV TORCH_CUDA_ARCH_LIST=Turing
ARG ACCESS_TOKEN=none
RUN python3 examples/05_stable_diffusion/compile.py --token $ACCESS_TOKEN


# python3 examples/05_stable_diffusion/demo.py --token $ACCESS_TOKEN --prompt "Mountain Rainier in van Gogh's world"
ENTRYPOINT ["python3", "examples/05_stable_diffusion/demo.py", "--token", "$ACCESS_TOKEN"] 
CMD "Mountain Rainier in van Gogh's world"


