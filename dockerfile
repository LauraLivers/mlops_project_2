FROM python:3.12-slim as base

# system dependencies for hardware detection
RUN apt-get update && apt-get install -y \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# hardware detection
FROM base as hardware-detector
COPY detect_hardware.py /detect_hardware.py
RUN python /detect_hardware.py > /hardware_info.txt

# main application stage
FROM base as app
COPY --from=hardware-detector /hardware_info.txt /hardware_info.txt

# working directory
WORKDIR /app

# requirements files depending on hardware
COPY requirements-base.txt requirements-base.txt 
COPY requirements-cuda.txt requirements-cuda.txt
COPY requirements-rocm.txt requirements-rocm.txt
COPY requirements-cpu.txt requirements-cpu.txt

# install dependencies based on detected files
RUN HARDWARE=$(cat /hardware_info.txt) && \
    echo "Detected hardware: $HARDWARE" && \
    if [ "$HARDWARE" = 'nvidia' ]; then \
        echo "Installing CUDA PyTorch" && \
        pip install -q -r requirements-cuda.txt&& \
        pip install -q -r requirements-base.txt; \
    elif [ "$HARDWARE" = 'amd' ]; then \
        echo "Installing ROCm PyTorch" && \
        pip install -q -r requirements-rocm.txt&& \
        pip install -q -r requirements-base.txt ; \
    else \
        echo "Installing CPU PyTorch" && \
        pip install -q -r requirements-cpu.txt&& \
        pip install -q -r requirements-base.txt; \
    fi

RUN if [ "$(cat /hardware_info.txt)" = "amd" ]; then \
    apt-get update && apt-get install -y prelink && \
    find /usr/local/lib/python3.12/site-packages -name "*.so*" -exec execstack -c {} \; 2>/dev/null || true && \
    rm -rf /var/lib/apt/lists/*; \
fi

# copy application code
COPY . .

# expose port
EXPOSE 8000

CMD ["python", "mlops_hp_2.py"]