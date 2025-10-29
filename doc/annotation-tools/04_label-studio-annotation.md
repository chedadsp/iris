# YOLO + Label Studio Setup Summary

## Table of Contents
1. [Overview](#overview)
2. [Part 1: Manual Annotation Workflow](#part-1-manual-annotation-workflow)
   - 1.1 [Label Studio Installation](#11-label-studio-installation)
   - 1.2 [Video Preparation](#12-video-preparation)
   - 1.3 [Project Setup](#13-project-setup)
   - 1.4 [Import Data](#14-import-data)
   - 1.5 [Manual Annotation Process](#15-manual-annotation-process)
3. [Part 2: AI-Assisted Annotation with YOLO](#part-2-ai-assisted-annotation-with-yolo)
   - 2.1 [Why Use Pre-labeling?](#21-why-use-pre-labeling)
   - 2.2 [YOLO Model Overview](#22-yolo-model-overview)
   - 2.3 [ML Backend Installation](#23-ml-backend-installation)
   - 2.4 [YOLO CLI Setup](#24-yolo-cli-setup)
   - 2.5 [Python Path Configuration](#25-python-path-configuration)
   - 2.6 [Launch ML Backend](#26-launch-ml-backend)
   - 2.7 [Connect Backend to Label Studio](#27-connect-backend-to-label-studio)
   - 2.8 [Using Pre-labeling](#28-using-pre-labeling)
   - 2.9 [Troubleshooting](#29-troubleshooting)
4. [References](#references)

---

## Overview
This document describes two annotation approaches in Label Studio:
1. **Manual Annotation** - Traditional hand-labeling of objects
2. **AI-Assisted Annotation** - Using YOLOv8 pre-labeling to speed up the process

---

## Part 1: Manual Annotation Workflow
=====================================

### 1.1 Label Studio Installation
- Installed Label Studio on Linux:
  ```bash
  pip install label-studio
  ```
- Launch Label Studio:
  ```bash
  label-studio start
  ```
- Access via: http://localhost:8080
- Create account on first run (email + password)

### 1.2 Video Preparation
- Convert video to individual frames for annotation:
  ```bash
  ffmpeg -i test_video.mp4 -vf fps=1 frames/frame_%04d.jpg
  ```
  - `fps=1` extracts 1 frame per second
  - Adjust fps value based on video content (higher for fast motion)
  - Output stored in `frames/` directory

### 1.3 Project Setup
- Create new project in Label Studio
- Project Type: **Computer Vision → Object Detection with Bounding Boxes**

### 1.4 Import Data
- Go to project → Import
- Upload all frames from `frames/` directory
- Supported formats: JPG, PNG, JPEG
- Images appear as tasks in the task list

### 1.5 Manual Annotation Process
1. **Open a task** from the task list
2. **Select label** (Car, Bus, Truck) from the toolbar
3. **Draw bounding box** by clicking and dragging on the image
4. **Adjust box** by dragging corners or edges
5. **Delete box** if incorrect (select and press Delete)
6. **Submit** annotation when complete
7. **Repeat** for all frames

---

## Part 2: AI-Assisted Annotation with YOLO
============================================

### 2.1 Why Use Pre-labeling?
- **Speed**: Automatically generate initial annotations
- **Efficiency**: Human only reviews/corrects instead of creating from scratch
- **Consistency**: AI provides uniform detection across frames
- **Use Case**: When you have many similar objects across frames

### 2.2 YOLO Model Overview
- Using **YOLOv8n** (nano) - fastest, good for real-time
- Pre-trained on COCO dataset (80 classes including vehicles)
- Can detect: car, bus, truck, motorcycle, bicycle, etc.
- Model will run in background and suggest bounding boxes

### 2.3 ML Backend Installation

#### Install Dependencies
```bash
# Label Studio ML backend
pip install label-studio-ml

# YOLOv8
pip install ultralytics

# Additional dependencies
pip install torch torchvision opencv-python
```

#### Clone Backend Repository
```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/label_studio_ml/examples/yolo
```

### 2.4 YOLO CLI Setup
- Add YOLO CLI to PATH if not accessible:
  ```bash
  export PATH=$PATH:~/.local/bin
  ```
- Test YOLO prediction standalone (optional):
  ```bash
  yolo predict model=yolov8n.pt source='frames/' save=True
  ```
  - Results saved to `runs/detect/predict/`

### 2.5 Python Path Configuration
- Fix module import errors by exporting Python paths:
  ```bash
  # Find your Python site-packages location
  python3 -c "import site; print(site.getsitepackages()[0])"
  
  # Export paths (replace /path/to/label-studio-ml-backend with your actual path)
  export PYTHONPATH=$PYTHONPATH:$(python3 -c "import site; print(site.getsitepackages()[0])")
  export PYTHONPATH=$PYTHONPATH:/path/to/label-studio-ml-backend
  ```
- Add to `~/.bashrc` for permanent effect:
  ```bash
  echo 'export PYTHONPATH=$PYTHONPATH:'$(python3 -c "import site; print(site.getsitepackages()[0])")'' >> ~/.bashrc
  echo 'export PYTHONPATH=$PYTHONPATH:/path/to/label-studio-ml-backend' >> ~/.bashrc
  source ~/.bashrc
  ```
  **Note**: Replace `/path/to/label-studio-ml-backend` with the actual directory where you cloned the repository
    ```

### 2.6 Launch ML Backend
```bash
cd ~/Downloads/test_video/label-studio-ml-backend/label_studio_ml/examples/yolo
label-studio-ml start .
```
- Backend runs on: **http://localhost:9090**
- Keep terminal open while using pre-labeling
- Logs show prediction requests in real-time

### 2.7 Connect Backend to Label Studio
1. Open Label Studio project
2. Go to **Settings → Machine Learning**
3. Click **Add Model**
4. Enter backend URL: `http://localhost:9090`
5. Click **Validate and Save**
6. Enable **Use for interactive preannotation**
7. Enable **Use for predictions when data is imported**

### 2.8 Using Pre-labeling
#### Interactive Prediction
- Open any task
- Click **Predict** button (or press `P`)
- YOLO generates bounding boxes automatically
- Review and correct predictions:
  - Adjust box positions/sizes
  - Delete incorrect detections
  - Add missing objects manually
  - Change labels if misclassified
- Submit corrected annotations

#### Batch Pre-labeling
- Select multiple tasks (checkbox)
- Click **Actions → Retrieve Predictions**
- Backend processes all selected tasks
- Review each task individually afterward

### 2.9 Troubleshooting

#### Backend Not Connecting
- Check if backend is running: `curl http://localhost:9090/health`
- Check firewall settings
- Verify URL in Label Studio matches backend address

---


## References
=============
- Label Studio Docs: https://labelstud.io/guide/
- YOLOv8 Docs: https://docs.ultralytics.com/
- ML Backend: https://github.com/HumanSignal/label-studio-ml-backend
````