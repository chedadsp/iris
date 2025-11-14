# Xtreme1 Annotation

## Table of Contents
1. [Overview](#1-overview)  
2. [What Xtreme1 needs to run your service](#2-what-xtreme1-needs-to-run-your-service)  
   - [Reachable HTTP endpoint](#21-reachable-http-endpoint)  
   - [Request / response contract](#22-request--response-contract)  
   - [Point-cloud storage access](#23-point-cloud-storage-access)  
   - [Model runtime environment](#24-model-runtime-environment)  
   - [Xtreme1 configuration](#25-xtreme1-configuration)  
   - [Networking & security](#26-networking--security)  
   - [Observability](#27-observability)  
3. [Xtreme1 Integration](#3-xtreme1-integration)  
   3.1. [What Xtreme1 expects](#31-what-xtreme1-expects)  
   3.2. [Your service (backend)](#32-your-service-backend)  
   3.3. [Xtreme1 settings](#33-xtreme1-settings)  
4. [Notes](#4-notes)

## 1. Overview
Xtreme1 is a platform for managing LiDAR datasets and running automatic annotation models (services) against uploaded point clouds. It calls your backend service for each PCD to obtain detections/annotations and integrates returned results into the platform UI and dataset metadata.

## 2. What Xtreme1 needs to run your service
- Reachable HTTP endpoint:
  - POST /pointCloud/recognition must be accessible from the Xtreme1 server (public IP, LAN IP, or via tunnel like ngrok).
- Standard request/response contract:
  - Xtreme1 sends JSON with items containing id and pointCloudUrl.
  - Your service returns the prescribed JSON structure with per-item objects, codes and messages.
- Accessible point-cloud storage:
  - PCD/LAS files must be downloadable by your service using the provided URLs.
- Model runtime environment:
  - Necessary dependencies (Python packages, ML libs), CPU/GPU as required, and sufficient RAM/disk.
- Configuration in Xtreme1:
  - Model registered in UI, Data Type set to Lidar, and Settings → URL pointing to your endpoint.
- Networking & security:
  - Open ports, TLS if required, and any auth tokens handled by the service or platform.
- Observability:
  - Logging and error responses for easy debugging and Test Connection support in the UI.

## 3. Xtreme1 Integration
Brief overview of how to add a model/service to Xtreme1 so it can perform annotation on uploaded PCDs.

### 3.1 What Xtreme1 expects
- Xtreme1 will call your backend once per PCD file with:
  POST http://<your_ip>:<your_port>/pointCloud/recognition
- Payload example:
  {
    "datas": [
      {
        "id": 1,
        "pointCloudUrl": "http://<xtreme1_storage>/point_cloud/frame_000001.pcd"
      }
    ]
  }

### 3.2 Your service (backend)
- Implement an HTTP endpoint that:
  - Receives the POST payload.
  - Downloads the PCD from pointCloudUrl (e.g., requests.get()).
  - Loads / processes the point cloud (or returns dummy results while model is not ready).
  - Returns JSON in Xtreme1 format, e.g.:
  ```python
    {
      "code": "OK",
      "message": "",
      "data": [
        {
          "id": 1,
          "code": 0,
          "message": "",
          "objects": [
            {
              "label": "CAR",
              "confidence": 0.92,
              "x": 10.0,
              "y": 5.0,
              "z": 0.0,
              "dx": 4.2,
              "dy": 2.1,
              "dz": 1.7,
              "rotX": 0.0,
              "rotY": 0.0,
              "rotZ": 0.3
            }
          ]
        }
      ]
    }
  ```

### 3.3 Xtreme1 settings
- In Xtreme1 Models → Create My Model (Data Type: Lidar).
- Edit model Settings → URL: set to:
  http://<your_host>:5000/pointCloud/recognition
- Click Test Connection — Xtreme1 should receive the expected JSON response.

## 4. Notes
- Ensure Xtreme1 can reach your backend: run on a reachable IP or use ngrok for temporary public URL.
- Log and return clear error codes/messages for debugging in Xtreme1.