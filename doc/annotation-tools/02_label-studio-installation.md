# Label Studio Installation Guide for Linux

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Verify Prerequisites](#verify-prerequisites)
- [Installation Steps](#installation-steps)
  - [Step 1: Install System Dependencies](#step-1-install-system-dependencies)
  - [Step 2: Install Label Studio](#step-2-install-label-studio)
  - [Step 3: Handle Missing Dependencies](#step-3-handle-missing-dependencies)
- [Project Initialization](#project-initialization)
  - [Create a New Project](#create-a-new-project)
  - [Start the Label Studio Server](#start-the-label-studio-server)
- [Initial Access and Configuration](#initial-access-and-configuration)
  - [First-Time Setup](#first-time-setup)
- [Additional Resources](#additional-resources)


## Overview
This guide provides comprehensive instructions for installing and configuring Label Studio on Linux systems. Label Studio is an open-source data labeling tool that supports various annotation tasks including image classification, object detection, and text annotation.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ or similar distribution)
- **Python**: Version 3.8 or higher
- **pip**: Python package installer

### Verify Prerequisites
Before proceeding with the installation, verify that Python and pip are properly installed:

```bash
python3 --version
pip3 --version
```

**Expected Output**:
- Python 3.8.x or higher
- pip 20.x or higher

---

## Installation Steps

### Step 1: Install System Dependencies

Install required system libraries for cryptographic operations and compilation:

```bash
sudo apt update
sudo apt install libffi-dev python3-dev build-essential -y
```

**Package Descriptions**:
- `libffi-dev`: Foreign Function Interface library development files
- `python3-dev`: Header files for building Python extensions
- `build-essential`: Essential compilation tools (gcc, make, etc.)

### Step 2: Install Label Studio

Install Label Studio via pip:

```bash
pip3 install label-studio
```

### Step 3: Handle Missing Dependencies

If you encounter module import errors, install the required Python packages:

```bash
pip3 install cffi cryptography async-timeout --force-reinstall
```

**Common Errors Resolved**:
- `ModuleNotFoundError: No module named 'async_timeout'`
- `ModuleNotFoundError: No module named '_cffi_backend'`

---

## Project Initialization

### Create a New Project

Initialize a new Label Studio project:

```bash
label-studio init vozilo-demo
```

**Project Structure Created**:
```
vozilo-demo/
├── config.json          # Project configuration
├── label_studio.db      # SQLite database
└── media/               # Uploaded files directory
```

### Start the Label Studio Server

Launch the Label Studio web interface:

```bash
label-studio start vozilo-demo
```

**Server Information**:
- Default URL: `http://localhost:8080`
- Default Port: 8080
- Configuration: Auto-loaded from project directory

---

## Initial Access and Configuration

### First-Time Setup

1. **Access the Web Interface**
   - Open your web browser
   - Navigate to: `http://localhost:8080`

2. **Create Admin Account**
   - Email: Enter your email address
   - Password: Choose a secure password
   - Confirm password

3. **Project Settings**
   - Project Name: Configure your project name
   - Description: Add project description (optional)

## Additional Resources

- **Official Documentation**: https://labelstud.io/guide/
- **GitHub Repository**: https://github.com/heartexlabs/label-studio
- **Community Forum**: https://slack.labelstud.io/

