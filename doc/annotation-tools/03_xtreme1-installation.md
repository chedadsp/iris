# Xtreme1 Installation Guide for Linux

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Required Software](#required-software)
  - [Verify Prerequisites](#verify-prerequisites)
- [Installation Steps](#installation-steps)
  - [Step 1: Install Docker and Docker Compose](#step-1-install-docker-and-docker-compose)
  - [Step 2: Configure Docker Service](#step-2-configure-docker-service)
  - [Step 3: Clone Xtreme1 Repository](#step-3-clone-xtreme1-repository)
  - [Step 4: Configure Docker Compose File](#step-4-configure-docker-compose-file)
  - [Step 5: Launch Xtreme1](#step-5-launch-xtreme1)
- [Accessing Xtreme1](#accessing-xtreme1)
  - [Web Interface](#web-interface)
- [Additional Resources](#additional-resources)
  - [Official Documentation](#official-documentation)
  - [Community Support](#community-support)
  - [Docker Resources](#docker-resources)

## Overview
Xtreme1 is a comprehensive data labeling and annotation platform designed for computer vision tasks. It provides advanced features for image annotation, 3D point cloud labeling, and dataset management. This guide provides step-by-step instructions for installing and configuring Xtreme1 on Linux systems using Docker.


## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ or similar distribution)
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Disk Space**: At least 20GB free space

### Required Software
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher

### Verify Prerequisites
Check if Docker and Docker Compose are installed:

```bash
docker --version
docker-compose --version
git --version
```

**Expected Output**:
- Docker version 20.10.x or higher
- Docker Compose version 1.29.x or higher
- Git version 2.x or higher

---

## Installation Steps

### Step 1: Install Docker and Docker Compose

Update system packages and install Docker:

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
```

**Package Descriptions**:
- `docker.io`: Docker container runtime engine
- `docker-compose`: Tool for defining and running multi-container Docker applications

### Step 2: Configure Docker Service

Start and enable Docker service to run on system boot:

```bash
sudo systemctl daemon-reload
sudo systemctl start docker
sudo systemctl enable docker
```

**Verify Docker Service Status**:
```bash
sudo systemctl status docker
```

**Expected Output**: Service should show as "active (running)"

### Step 3: Clone Xtreme1 Repository

Clone the official Xtreme1 repository from GitHub:

```bash
git clone https://github.com/xtreme1-io/xtreme1.git
cd xtreme1
```

### Step 4: Configure Docker Compose File

The repository contains a `docker-compose.yml` file that needs to be modified for compatibility.

#### Fix MinIO Image Version

Edit the `docker-compose.yml` file to update the MinIO image version:

```bash
nano docker-compose.yml
```

**Find and Replace**:
```yaml
# Original (may cause error)
image: bitnami/minio:2022.9.1

# Replace with
image: minio/minio:RELEASE.2024-09-07T16-13-09Z
```

**Why This Change?**:
- The Bitnami MinIO image version 2022.9.1 is no longer available in the registry
- Using the official MinIO image ensures compatibility and availability

### Step 5: Launch Xtreme1

Start all Xtreme1 services using Docker Compose:

```bash
docker-compose up
```

## Accessing Xtreme1

### Web Interface

1. **Open Web Browser**
   - Navigate to: `http://localhost:8190`
   
2. **Default Credentials**
   - Username: `admin@xtreme1.io`
   - Password: `admin123`
   
3. **First Login**
   - Change the default password immediately
   - Configure your profile settings


## Additional Resources

### Official Documentation
- **Xtreme1 Documentation**: https://docs.xtreme1.io/
- **GitHub Repository**: https://github.com/xtreme1-io/xtreme1
- **API Documentation**: https://docs.xtreme1.io/api/

### Community Support
- **Discord Community**: https://discord.gg/xtreme1
- **GitHub Issues**: https://github.com/xtreme1-io/xtreme1/issues
- **Stack Overflow**: Tag `xtreme1`

### Docker Resources
- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---