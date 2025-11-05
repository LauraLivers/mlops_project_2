FROM python:3.12-slim AS base
# system dependencies for hardware detection
RUN apt-get update && apt-get install -y \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# automatic hardware detection
FROM base AS hardware-detector
COPY detect_hardware.py /detect_hardware.py
RUN python /detect_hardware.py > /hardware_info.txt

# main application stage
FROM base AS app
COPY --from=hardware-detector /hardware_info.txt /hardware_info.txt

# working directory
WORKDIR /app

# requirements files depending on hardware
COPY requirements-base.txt requirements-base.txt 
COPY requirements-cuda.txt requirements-cuda.txt
COPY requirements-rocm.txt requirements-rocm.txt
COPY requirements-cpu.txt requirements-cpu.txt

# install dependencies based on detected files
RUN --mount=type=cache,target=/root/.cache/pip \
    HARDWARE=$(cat /hardware_info.txt) && \
    echo "Detected hardware: $HARDWARE" && \
    if [ "$HARDWARE" = 'nvidia' ]; then \
        echo "Installing CUDA PyTorch" && \
        pip install -r requirements-cuda.txt && \
        pip install -r requirements-base.txt; \
    elif [ "$HARDWARE" = 'amd' ]; then \
        echo "Installing ROCm PyTorch" && \
        pip install -r requirements-rocm.txt && \
        pip install -r requirements-base.txt; \
    else \
        echo "Installing CPU PyTorch" && \
        pip install -r requirements-cpu.txt && \
        pip install -r requirements-base.txt; \
    fi

# Fix ROCm for AMD
RUN if [ "$(cat /hardware_info.txt)" = "amd" ]; then \
        apt-get update && apt-get install -y pax-utils && \
        find /usr/local/lib/python3.12/site-packages/torch/lib -name "*.so*" -exec execstack -c {} \; && \
        find /usr/local/lib/python3.12/site-packages -name "*hip*.so*" -exec execstack -c {} \; && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# copy application code
COPY . .

# expose port
EXPOSE 8000

CMD ["python", "mlops_hp_2.py"]