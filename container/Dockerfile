# Image for SageMaker Processing jobs. Build from the repository root, for the platform SageMaker
# uses (x86_64) and, if wanted, arm64 for local runs (byte-identical on the golden files):
#   scripts/push_image.sh                                  # builds linux/amd64 and pushes to ECR
#   docker build --platform linux/amd64 -f container/Dockerfile -t twin:dev .
#
# Stage 1 builds the engine with the reference compiler (the same pinned GCC image as
# engine/tools/ref.sh) and the reference flags (-O2 -ffp-contract=off). Stage 2 is a slim Python
# image with the twin package, pinned dependencies, the engine binary and the GCC 13.3 runtime
# libraries it was linked against.

FROM gcc:13.3.0-bookworm@sha256:1d71f0f3450214bef38fe09e6f610fb6cca90cf97b43f4ce845bfc32a4168818 AS engine
WORKDIR /src
COPY engine/ /src/
RUN make cli BUILD=/out && /out/iirabm_cli --version \
    && mkdir -p /out/lib && cp -P /usr/local/lib64/libstdc++.so.6* /usr/local/lib64/libgcc_s.so.1 /out/lib/

FROM python:3.11-slim-bookworm
ARG GIT_SHA=unknown
ARG ENGINE_COMMIT=unknown
LABEL org.opencontainers.image.title="iirabm-twin" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.iirabm-twin.engine-commit="${ENGINE_COMMIT}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY --from=engine /out/iirabm_cli /usr/local/bin/iirabm_cli
COPY --from=engine /out/lib/ /usr/local/lib64/
ENV LD_LIBRARY_PATH=/usr/local/lib64 IIRABM_BIN=/usr/local/bin/iirabm_cli
RUN iirabm_cli --version

COPY container/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md LICENSE /opt/twin/
COPY src /opt/twin/src
COPY config/prior.yaml config/measurement.yaml /opt/twin/config/
RUN pip install --no-deps /opt/twin

# Runs as root on purpose: SageMaker mounts /opt/ml/processing/* root-owned, so a non-root user
# cannot write job output. Jobs run with network isolation and carry no credentials. SageMaker
# sets the entrypoint per job.
WORKDIR /opt/ml
