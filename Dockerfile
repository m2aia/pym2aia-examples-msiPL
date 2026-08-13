FROM ghcr.io/m2aia/python-m2aia-cuda:latest

# Build arguments
ARG BUILD_DATE
ARG BUILD_VERSION

# Labels
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.version=$BUILD_VERSION
LABEL org.label-schema.name="m2aia/msiPL"
LABEL org.label-schema.description="msiPL Peak Learning: deep learning for mass spectrometry imaging (GPU)"
LABEL org.label-schema.url="https://m2aia.de/"
LABEL org.label-schema.vendor="m2aia.de"

# Install TensorFlow (protobuf/numpy/scipy pinned for compatibility: TF
# 2.17.0 requires numpy<2, so numpy and scipy must be resolved together
# with it here rather than relying on the newer, unpinned versions already
# installed by the base image)
RUN pip install --no-cache-dir \
    protobuf==4.25.3 \
    "numpy<2" \
    "scipy<1.14" \
    tensorflow[and-cuda]==2.17.0

COPY msiPL /msiPL
COPY app_msiPL.py /app_msiPL.py

ENTRYPOINT [ "python", "/app_msiPL.py" ]
