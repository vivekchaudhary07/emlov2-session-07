# Stage 1: Builder/Compiler
FROM python:3.9-slim-buster AS build
 
RUN apt-get update -y && apt install -y --no-install-recommends git\
    && pip install --no-cache-dir -U pip

COPY requirements.txt .

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH


RUN pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.10.2%2Bcpu-cp39-cp39-linux_x86_64.whl \
    && pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torchvision-0.11.3%2Bcpu-cp39-cp39-linux_x86_64.whl \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim-buster 

COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /code

COPY . .

# ENTRYPOINT ["/bin/bash"]



