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
# nable to find image 'python:3.12-slim' locally
# 3.12-slim: Pulling from library/python
# 38513bd72563: Already exists 
# f2a111092025: Already exists 
# 79f2dc6dd7d8: Already exists 
# d2876f169c02: Already exists 
# Digest: sha256:e97cf9a2e84d604941d9902f00616db7466ff302af4b1c3c67fb7c522efa8ed9
# Status: Downloaded newer image for python:3.12-slim
# Hit:1 http://deb.debian.org/debian trixie InRelease
# Get:2 http://deb.debian.org/debian trixie-updates InRelease [47.3 kB]
# Get:3 http://deb.debian.org/debian-security trixie-security InRelease [43.4 kB]
# Get:4 http://deb.debian.org/debian trixie/main amd64 Packages [9669 kB]
# Get:5 http://deb.debian.org/debian trixie-updates/main amd64 Packages [5412 B]
# Get:6 http://deb.debian.org/debian-security trixie-security/main amd64 Packages [69.3 kB]
# Fetched 9834 kB in 1s (7019 kB/s)
# Reading package lists...
# Reading package lists...
# Building dependency tree...
# Reading state information...
# The following additional packages will be installed:
#   libkmod2 libpci3 pci.ids
# Suggested packages:
#   bzip2 wget | curl | lynx
# The following NEW packages will be installed:
#   libkmod2 libpci3 pci.ids pciutils
# 0 upgraded, 4 newly installed, 0 to remove and 0 not upgraded.
# Need to get 537 kB of archives.
# After this operation, 2111 kB of additional disk space will be used.
# Get:1 http://deb.debian.org/debian trixie/main amd64 libkmod2 amd64 34.2-2 [63.2 kB]
# Get:2 http://deb.debian.org/debian trixie/main amd64 pci.ids all 0.0~2025.06.09-1 [269 kB]
# Get:3 http://deb.debian.org/debian trixie/main amd64 libpci3 amd64 1:3.13.0-2 [75.6 kB]
# Get:4 http://deb.debian.org/debian trixie/main amd64 pciutils amd64 1:3.13.0-2 [129 kB]
# debconf: unable to initialize frontend: Dialog
# debconf: (TERM is not set, so the dialog frontend is not usable.)
# debconf: falling back to frontend: Readline
# debconf: unable to initialize frontend: Readline
# debconf: (Can't locate Term/ReadLine.pm in @INC (you may need to install the Term::ReadLine module) (@INC entries checked: /etc/perl /usr/local/lib/x86_64-linux-gnu/perl/5.40.1 /usr/local/share/perl/5.40.1 /usr/lib/x86_64-linux-gnu/perl5/5.40 /usr/share/perl5 /usr/lib/x86_64-linux-gnu/perl-base /usr/lib/x86_64-linux-gnu/perl/5.40 /usr/share/perl/5.40 /usr/local/lib/site_perl) at /usr/share/perl5/Debconf/FrontEnd/Readline.pm line 8, <STDIN> line 4.)
# debconf: falling back to frontend: Teletype
# debconf: unable to initialize frontend: Teletype
# debconf: (This frontend requires a controlling tty.)
# debconf: falling back to frontend: Noninteractive
# Fetched 537 kB in 0s (1652 kB/s)
# Selecting previously unselected package libkmod2:amd64.
# (Reading database ... 5644 files and directories currently installed.)
# Preparing to unpack .../libkmod2_34.2-2_amd64.deb ...
# Unpacking libkmod2:amd64 (34.2-2) ...
# Selecting previously unselected package pci.ids.
# Preparing to unpack .../pci.ids_0.0~2025.06.09-1_all.deb ...
# Unpacking pci.ids (0.0~2025.06.09-1) ...
# Selecting previously unselected package libpci3:amd64.
# Preparing to unpack .../libpci3_1%3a3.13.0-2_amd64.deb ...
# Unpacking libpci3:amd64 (1:3.13.0-2) ...
# Selecting previously unselected package pciutils.
# Preparing to unpack .../pciutils_1%3a3.13.0-2_amd64.deb ...
# Unpacking pciutils (1:3.13.0-2) ...
# Setting up pci.ids (0.0~2025.06.09-1) ...
# Setting up libpci3:amd64 (1:3.13.0-2) ...
# Setting up libkmod2:amd64 (34.2-2) ...
# Setting up pciutils (1:3.13.0-2) ...
# Processing triggers for libc-bin (2.41-12) ...
# 00:00.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Root Complex
# 00:00.2 IOMMU: Advanced Micro Devices, Inc. [AMD] Phoenix IOMMU
# 00:01.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Host Bridge
# 00:02.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Host Bridge
# 00:02.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix GPP Bridge
# 00:02.2 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix GPP Bridge
# 00:02.4 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix GPP Bridge
# 00:03.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Host Bridge
# 00:03.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Family 19h USB4/Thunderbolt PCIe tunnel
# 00:04.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Host Bridge
# 00:04.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Family 19h USB4/Thunderbolt PCIe tunnel
# 00:08.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Host Bridge
# 00:08.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Internal GPP Bridge to Bus [C:A]
# 00:08.2 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Internal GPP Bridge to Bus [C:A]
# 00:08.3 PCI bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Internal GPP Bridge to Bus [C:A]
# 00:14.0 SMBus: Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller (rev 71)
# 00:14.3 ISA bridge: Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge (rev 51)
# 00:18.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 0
# 00:18.1 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 1
# 00:18.2 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 2
# 00:18.3 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 3
# 00:18.4 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 4
# 00:18.5 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 5
# 00:18.6 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 6
# 00:18.7 Host bridge: Advanced Micro Devices, Inc. [AMD] Phoenix Data Fabric; Function 7
# 01:00.0 Ethernet controller: Realtek Semiconductor Co., Ltd. RTL8111/8168/8211/8411 PCI Express Gigabit Ethernet Controller (rev 0e)
# 02:00.0 Network controller: Qualcomm Technologies, Inc QCNFA765 Wireless Network Adapter (rev 01)
# 03:00.0 Non-Volatile memory controller: KIOXIA Corporation NVMe SSD Controller XG8 (rev 01)
# c4:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Phoenix3 (rev d0)
# c4:00.1 Audio device: Advanced Micro Devices, Inc. [AMD/ATI] Radeon High Definition Audio Controller [Rembrandt/Strix]
# c4:00.2 Encryption controller: Advanced Micro Devices, Inc. [AMD] Phoenix CCP/PSP 3.0 Device
# c4:00.3 USB controller: Advanced Micro Devices, Inc. [AMD] Device 15b9
# c4:00.4 USB controller: Advanced Micro Devices, Inc. [AMD] Device 15ba
# c4:00.5 Multimedia controller: Advanced Micro Devices, Inc. [AMD] Audio Coprocessor (rev 63)
# c4:00.6 Audio device: Advanced Micro Devices, Inc. [AMD] Family 17h/19h/1ah HD Audio Controller
# c5:00.0 Non-Essential Instrumentation [1300]: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Function
# c5:00.1 Signal processing controller: Advanced Micro Devices, Inc. [AMD] AMD IPU Device
# c6:00.0 Non-Essential Instrumentation [1300]: Advanced Micro Devices, Inc. [AMD] Phoenix Dummy Function
# c6:00.3 USB controller: Advanced Micro Devices, Inc. [AMD] Device 15c0
# c6:00.4 USB controller: Advanced Micro Devices, Inc. [AMD] Device 15c1
# c6:00.5 USB controller: Advanced Micro Devices, Inc. [AMD] Pink Sardine USB4/Thunderbolt NHI controller #1
# c6:00.6 USB controller: Advanced Micro Devices, Inc. [AMD] Pink Sardine USB4/Thunderbolt NHI controller #2