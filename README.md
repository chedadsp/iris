# IRIS - Integrated Roadside Intelligence System

A comprehensive Python-based LIDAR point cloud processing and visualization system for analyzing roadside vehicle scans. IRIS processes E57, PCD, and ROS bag files to detect vehicles, extract interior spaces, perform sensor fusion with camera data, and enable interactive 3D analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## Features

### Core Capabilities
- **Multi-Format Point Cloud Loading**: E57, PCD, and ROS bag file support
- **Vehicle Detection & Analysis**: Automated vehicle identification using DBSCAN clustering
- **Interior Extraction**: 3D occupancy grid-based interior space detection
- **Interactive 3D Visualization**: PyVista/VTK-based real-time rendering
- **Sensor Fusion**: Camera-LIDAR fusion with YOLO object detection
- **Vehicle Tracking**: Multi-frame tracking with trajectory analysis
- **Human Model Positioning**: Interactive human model placement in point cloud scenes

### Analysis Pipeline
- Ground plane separation (RANSAC-based)
- Vehicle clustering and identification
- Cockpit and dashboard detection
- Interior point extraction
- Spatial analysis with voxel grids

### User Interfaces
- **GUI Mode**: Full-featured Tkinter interface
- **CLI Mode**: Interactive command-line interface
- **Headless Mode**: Automated batch processing

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Application Modes](#application-modes)
- [Sensor Fusion & Tracking](#sensor-fusion--tracking)
- [Demo Applications](#demo-applications)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### Prerequisites

- **Python 3.11** (required - Python 3.12+ not yet supported)
- **Poetry** package manager
- **macOS, Linux, or Windows** (macOS requires special VTK handling)

### Step 1: Install Poetry

Poetry is used for dependency management. Install it using one of these methods:

#### macOS/Linux
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Windows (PowerShell)
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

#### Alternative: pip
```bash
pip install poetry
```

Verify installation:
```bash
poetry --version
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/yourusername/iris.git
cd iris
```

### Step 3: Install Dependencies

#### Basic Installation
```bash
# Install core dependencies
poetry install
```

#### Optional Features
```bash
# ROS bag file support
poetry install -E ros

# PointTransformerV3 deep learning support
poetry install -E ptv3

# MMDetection3D support (not compatible with Apple Silicon)
poetry install -E mmdet3d

# Install all optional dependencies
poetry install -E all
```

### Step 4: Activate the Environment

```bash
# Activate the Poetry virtual environment
poetry shell
```

Or run commands directly:
```bash
poetry run python src/launcher.py
```

## Quick Start

### 1. Launch the GUI Application

```bash
python src/launcher.py
```

This opens the main graphical interface where you can:
- Load point cloud files (E57, PCD, ROS bag)
- Run vehicle analysis
- Visualize results in 3D
- Perform interactive cube selection
- Position human models

### 2. Process a Single File (Headless)

```bash
python src/launcher.py --analysis-only data/your_file.e57
```

### 3. Interactive CLI Mode

```bash
python src/launcher.py --mode cli
```

### 4. macOS Users (Recommended)

For optimal VTK stability on macOS:
```bash
python src/macos_launcher.py
```

## Usage

### Application Modes

#### GUI Mode (Default)

The comprehensive graphical interface with tabs for:
- **File Operations**: Load and preview point cloud files
- **Vehicle Analysis**: Automated processing pipeline
- **Cube Selection**: Interactive 3D region selection
- **Human Positioning**: Position human models in scenes
- **Results**: View and export analysis results

```bash
# Launch GUI (all equivalent)
python src/launcher.py
python src/launcher.py --mode gui
poetry run python src/launcher.py
```

#### CLI Mode

Interactive command-line interface for terminal-based workflows:

```bash
python src/launcher.py --mode cli
```

Features:
- File selection and loading
- Analysis execution
- Parameter configuration
- Result inspection

#### Headless Mode

Automated processing without GUI (ideal for batch processing):

```bash
# Process a single file
python src/launcher.py --analysis-only path/to/file.e57

# Full headless mode with custom parameters
python src/launcher.py --mode headless --input data/scan.e57 --output results/
```

### Command-Line Arguments

```bash
python src/launcher.py [OPTIONS]

Options:
  --mode {gui,cli,headless}    Launch mode (default: gui)
  --cli                        Launch CLI mode (shortcut)
  --analysis-only FILE         Headless analysis of single file
  --input PATH                 Input file path
  --output PATH               Output directory
  --cube-editor               Launch cube selection tool
  --human-positioner          Launch human positioning tool
  --help                      Show help message
```

## Sensor Fusion & Tracking

IRIS includes advanced sensor fusion capabilities for camera-LIDAR integration.

### Sensor Fusion Pipeline

Combines YOLO object detection with LIDAR point clouds:

```bash
# Basic sensor fusion
python src/sensor_fusion.py --sequence seq-53 --device 105

# With visualization
python src/sensor_fusion.py --sequence seq-53 --device 105 --visualize

# Process all frames
python src/run_all_sensor_fusion.py --sequence seq-53
```

**Features:**
- YOLO-based car detection in camera images
- Camera-LIDAR calibration and projection
- 3D bounding box generation
- Point cloud association with detections

### Vehicle Tracking

Multi-frame tracking with trajectory analysis:

```bash
# Basic tracking
python src/car_tracking.py --sequence seq-53 --device 105

# Enhanced tracking with 3D models
python src/enhanced_car_tracking.py --sequence seq-53 --max_frames 10

# Temporal analysis
python src/demo_temporal_analysis.py
```

**Features:**
- Multi-frame vehicle tracking
- Trajectory analysis
- Speed estimation
- 3D vehicle model construction
- Movement pattern analysis

### Quick Tests

```bash
# Quick sensor fusion test
python src/quick_sensor_fusion_test.py

# Batch processing
python src/simple_batch_fusion.py
```

## Demo Applications

### Passenger Detection Demo

Demonstrates interior analysis and human detection:

```bash
# Interactive demo launcher
python src/launch_passenger_demo.py

# Standalone demo
python src/passenger_demo_standalone.py

# Animated demonstration
python src/passenger_detection_animation.py
```

### Sales Deck Visualizations

Generate professional visualizations for presentations:

```bash
python src/sales_deck_launcher.py --sequence seq-53 --device 105 --max_frames 8
```

Creates:
- LIDAR scene overviews
- Vehicle detection visualizations
- Tracking trajectory plots
- 3D model reconstructions

### Sequence Analysis

Analyze temporal sequences of LIDAR scans:

```bash
python src/sequence_lidar_analysis.py --sequence seq-53
python src/sequence_fusion.py --sequence seq-53 --device 105
```

## Project Structure

```
iris/
├── src/
│   ├── launcher.py              # Universal launcher (entry point)
│   ├── macos_launcher.py        # macOS-optimized launcher
│   ├── lidar_gui_app.py         # Legacy GUI application
│   │
│   ├── core/                    # Core launcher modules
│   │   ├── gui_launcher.py      # GUI mode implementation
│   │   ├── cli_launcher.py      # CLI mode implementation
│   │   ├── headless_launcher.py # Headless mode implementation
│   │   ├── launcher_base.py     # Base launcher interface
│   │   ├── dependency_checker.py
│   │   └── environment_setup.py
│   │
│   ├── point_cloud/             # Point cloud processing
│   │   ├── loaders/             # File format loaders
│   │   ├── processors/          # Processing stages
│   │   ├── pipeline/            # Analysis pipeline
│   │   ├── services/            # High-level services
│   │   ├── config.py            # Configuration constants
│   │   ├── vtk_utils.py         # VTK safety management
│   │   └── error_handling.py    # Error handling
│   │
│   ├── platforms/               # Platform-specific code
│   │
│   ├── sensor_fusion.py         # Camera-LIDAR fusion
│   ├── car_tracking.py          # Vehicle tracking
│   ├── enhanced_car_tracking.py # Advanced tracking
│   └── [demo scripts...]        # Various demo applications
│
├── data/                        # Data files and datasets
│   └── README.md               # Dataset references
│
├── pyproject.toml              # Poetry dependencies
├── workspace.dsl               # Architecture documentation (Structurizr)
└── yolov8n.pt                 # YOLO model weights
```

## Configuration

### Analysis Parameters

Edit `src/point_cloud/config.py` to adjust:

```python
class AnalysisConfig:
    # Ground separation
    GROUND_HEIGHT_THRESHOLD = 0.5  # meters
    GROUND_PLANE_TOLERANCE = 0.1   # meters

    # Vehicle identification
    DBSCAN_EPS = 0.3              # clustering radius
    DBSCAN_MIN_SAMPLES = 50       # minimum cluster size
    VEHICLE_MAX_HEIGHT = 3.0      # meters

    # Interior detection
    GRID_RESOLUTION = 0.1         # voxel size (10cm)
    INTERIOR_THRESHOLD_3D = 2     # distance threshold
```

### Visualization Settings

```python
class RenderConfig:
    POINT_SIZE = 2.0
    BACKGROUND_COLOR = (0.1, 0.1, 0.1)
    WINDOW_SIZE = (1920, 1080)
    CAMERA_POSITION = 'xy'
```

## Datasets

IRIS supports various public roadside LIDAR datasets. See [data/README.md](data/README.md) for:

- **DAIR-V2X** (China): Vehicle-infrastructure cooperative dataset
- **TAMU** (USA): Roadside LIDAR dataset
- **A9 Intersection** (Germany): Providentia dataset
- **DAIR-RCooper** (China): Roadside cooperative dataset
- **FLIR Thermal**: Thermal camera datasets

Place your dataset files in the `data/` directory.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest src/point_cloud/test_service_separation.py

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Architecture Documentation

The project includes comprehensive C4 model architecture diagrams in Structurizr DSL format:

- `workspace.dsl` - Complete system architecture
- `point_cloud_architecture.dsl` - Point cloud module details
- `point_cloud_processing_pipeline.dsl` - Processing pipeline

View these using [Structurizr Lite](https://structurizr.com/help/lite) or the online viewer.

## Troubleshooting

### Common Issues

#### VTK Segmentation Faults (macOS)

**Symptom**: Application crashes with segmentation fault during visualization

**Solutions**:
1. Use the macOS launcher: `python src/macos_launcher.py`
2. Ensure VTK environment is properly initialized
3. Check that all VTK resources are cleaned up properly

#### Python Version Issues

**Symptom**: Installation fails or dependencies conflict

**Solution**: Ensure you're using Python 3.11:
```bash
python --version  # Should show 3.11.x
poetry env use python3.11
poetry install
```

#### Missing Dependencies

**Symptom**: Import errors for optional packages

**Solution**: Install the appropriate extras:
```bash
poetry install -E all  # Install all optional dependencies
```

#### OpenCV Compatibility

**Symptom**: OpenCV errors or crashes

**Solution**: The project pins OpenCV to `<4.10` for compatibility. If you have a newer version globally, ensure you're using the Poetry environment:
```bash
poetry shell
python -c "import cv2; print(cv2.__version__)"
```

#### Memory Issues with Large Point Clouds

**Symptom**: Out of memory errors

**Solutions**:
1. Increase downsampling in preprocessing
2. Process files in headless mode
3. Adjust `GRID_RESOLUTION` in config for coarser voxels

### Platform-Specific Notes

#### macOS
- Use `src/macos_launcher.py` for best stability
- VTK requires special environment variables (handled automatically)
- Some deep learning features may not work on Apple Silicon

#### Linux
- Ensure graphics drivers are up to date for VTK rendering
- May need to install system packages for VTK/OpenGL

#### Windows
- VTK rendering may require additional DirectX configuration
- Use WSL2 for better compatibility if issues arise

### Getting Help

1. Check the [CLAUDE.md](CLAUDE.md) developer documentation
2. Review architecture diagrams in `.dsl` files
3. Examine test files for usage examples
4. Open an issue on GitHub

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style (Black formatting)
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyVista](https://pyvista.org/) for 3D visualization
- Uses [scikit-learn](https://scikit-learn.org/) for clustering algorithms
- YOLO object detection via [Ultralytics](https://ultralytics.com/)
- Point cloud processing with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- E57 file support via [pye57](https://github.com/davidcaron/pye57)

## Citation

If you use IRIS in your research, please cite:

```bibtex
@software{iris2025,
  title = {IRIS: Integrated Roadside Intelligence System},
  author = {Dimitrije Stojanovic},
  year = {2025},
  url = {https://github.com/yourusername/iris}
}
```

---

**Status**: Active Development | **Version**: 0.1.0 | **Updated**: 2025
