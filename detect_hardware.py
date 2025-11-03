import subprocess
import sys

def detect_gpu():
    """ detect GPU hardware and return appropriate backend """
    try:
       
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        lspci_output = result.stdout.lower()
        # NVIDIA / MPS
        if 'nvidia' in lspci_output and 'vga' in lspci_output:
            return 'nvidia'
        # AMD Ryzen
        elif 'amd' in lspci_output or 'ati' in lspci_output or 'radeon' in lspci_output:
            return 'amd'
        else:
            return 'cpu'
    except Exception:
        return 'cpu'

if __name__ == "__main__":
    hardware = detect_gpu()
    print(hardware)

# docker run --rm python:3.12-slim sh -c "apt-get update && apt-get install -y pciutils && lspci"