# Stage 1: Builder/Compiler
FROM python:3.7-slim-buster AS build

COPY requirements.txt .

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip3 install --no-cache-dir -U pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.7-slim-buster 

COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH
ENV GRADIO_SERVER_PORT 8080

WORKDIR /code

COPY configs/ configs/
COPY checkpoints/model.traced.pt checkpoints/
COPY src/ src/
COPY pyproject.toml .

EXPOSE 8080
ENTRYPOINT ["python3", "src/demo_trace.py"]

