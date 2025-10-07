workspace "LIDAR Point Cloud Processing System" "Comprehensive architecture documentation for LIDAR point cloud analysis and visualization application" {

    !identifiers hierarchical

    model {
        # External Systems
        user = person "LIDAR Analyst" "Research scientist or engineer analyzing vehicle LIDAR data" "User"
        fileSystem = softwareSystem "File System" "Local file storage for LIDAR data and analysis results" "External"

        # Core LIDAR Processing System
        lidarSystem = softwareSystem "LIDAR Point Cloud Processing System" "Comprehensive suite for processing, analyzing, and visualizing LIDAR point cloud data from vehicle scans" "System" {

            # User Interface Layer
            launcherApp = container "Universal Launcher" "Entry point application providing multi-mode access to all system features" "Python" "Application"
            guiApp = container "GUI Application" "Comprehensive Tkinter-based graphical interface for interactive LIDAR analysis" "Python/Tkinter" "GUI"
            macosLauncher = container "macOS Launcher" "Platform-optimized launcher with VTK environment configuration for macOS" "Python" "Platform"

            # Core Processing Layer
            vehicleAnalyzer = container "Vehicle Analysis Engine" "Main processing pipeline for E57 vehicle analysis with interior extraction" "Python/scikit-learn" "Engine"
            fileProcessor = container "Multi-Format File Processor" "Universal point cloud file reader supporting E57, PCD, and ROS bag formats" "Python/pye57" "Processor"
            visualizationEngine = container "3D Visualization Engine" "PyVista-based 3D rendering with VTK safety management" "Python/PyVista/VTK" "Renderer"

            # Analysis Modules
            interactiveSelector = container "Interactive Cube Selector" "3D cube-based region selection tool with real-time visual feedback" "Python/VTK" "Tool"
            humanPositioner = container "Human Model Positioner" "Interactive tool for positioning human models within point cloud scenes" "Python/VTK" "Tool"
            spatialAnalyzer = container "Spatial Analysis Module" "3D spatial analysis with voxel grids and distance transforms" "Python/SciPy" "Analyzer"

            # Utility and Support Layer
            vtkSafetyManager = container "VTK Safety Manager" "Centralized VTK resource management preventing segmentation faults" "Python/VTK" "Utility"
            errorHandler = container "Error Handling System" "Standardized error handling and logging with custom exceptions" "Python" "Support"
            configManager = container "Configuration Manager" "Centralized configuration for analysis parameters and rendering settings" "Python" "Config"
            outputManager = container "Output Management System" "Result saving and loading with metadata management" "Python/NumPy" "Storage"

            # Data Flow Relationships
            user -> launcherApp "Launches application"
            user -> guiApp "Interacts with GUI"

            # Launcher relationships
            launcherApp -> guiApp "Launches GUI mode"
            launcherApp -> vehicleAnalyzer "Launches headless analysis"
            launcherApp -> interactiveSelector "Launches cube editor"
            launcherApp -> humanPositioner "Launches human positioner"
            macosLauncher -> launcherApp "Provides macOS optimizations"

            # GUI relationships
            guiApp -> fileProcessor "Loads point cloud files"
            guiApp -> vehicleAnalyzer "Executes analysis pipeline"
            guiApp -> visualizationEngine "Creates 3D visualizations"
            guiApp -> interactiveSelector "Performs cube selection"
            guiApp -> humanPositioner "Positions human models"
            guiApp -> outputManager "Saves/loads results"

            # Core processing relationships
            vehicleAnalyzer -> fileProcessor "Reads LIDAR data"
            vehicleAnalyzer -> spatialAnalyzer "Performs spatial analysis"
            vehicleAnalyzer -> outputManager "Saves analysis results"
            vehicleAnalyzer -> visualizationEngine "Visualizes results"

            # File processing relationships
            fileProcessor -> fileSystem "Reads E57/PCD/BAG files"
            fileProcessor -> errorHandler "Handles file format errors"

            # Visualization relationships
            visualizationEngine -> vtkSafetyManager "Uses VTK safety wrappers"
            interactiveSelector -> vtkSafetyManager "Uses VTK safety wrappers"
            humanPositioner -> vtkSafetyManager "Uses VTK safety wrappers"

            # Support system relationships
            vehicleAnalyzer -> configManager "Uses analysis parameters"
            visualizationEngine -> configManager "Uses rendering settings"
            interactiveSelector -> configManager "Uses GUI parameters"
            humanPositioner -> configManager "Uses model parameters"

            outputManager -> fileSystem "Writes analysis results"
            errorHandler -> fileSystem "Writes log files"

            # Cross-cutting concerns
            guiApp -> errorHandler "Logs errors and status"
            vehicleAnalyzer -> errorHandler "Handles processing errors"
            fileProcessor -> errorHandler "Reports file errors"
            visualizationEngine -> errorHandler "Reports rendering errors"
        }

        # External dependencies and data sources
        e57Files = softwareSystem "E57 LIDAR Files" "Industry-standard LIDAR data files containing vehicle scan information" "External"
        pcdFiles = softwareSystem "PCD Files" "Point Cloud Data format files for 3D point cloud storage" "External"
        rosBagFiles = softwareSystem "ROS Bag Files" "Robot Operating System bag files containing sensor data streams" "External"

        # External library dependencies (represented as external systems)
        vtkLibrary = softwareSystem "VTK/PyVista Library" "Visualization Toolkit for 3D rendering and interaction" "External"
        scikitLearnLibrary = softwareSystem "scikit-learn Library" "Machine learning library for clustering and analysis algorithms" "External"
        numpyLibrary = softwareSystem "NumPy/SciPy Stack" "Scientific computing libraries for numerical operations" "External"

        # External system to system relationships
        e57Files -> lidarSystem "Provides vehicle LIDAR data"
        pcdFiles -> lidarSystem "Provides point cloud data"
        rosBagFiles -> lidarSystem "Provides ROS sensor data"
        lidarSystem -> vtkLibrary "Uses for 3D rendering"
        lidarSystem -> scikitLearnLibrary "Uses DBSCAN clustering"
        lidarSystem -> numpyLibrary "Uses numerical operations"
    }

    views {
        # System Context View
        systemContext lidarSystem "SystemContext" {
            title "LIDAR Point Cloud Processing System - System Context"
            description "High-level view showing the LIDAR processing system and its interactions with users and external systems"
            include *
            exclude "relationship.tag==Internal"
            autoLayout lr
        }

        # Container View - Overall Architecture
        container lidarSystem "ContainerView" {
            title "LIDAR Point Cloud Processing System - Container Architecture"
            description "Container-level architecture showing the main applications, processing engines, and support systems"
            include *
            exclude "element.tag==External"
            autoLayout tb
        }

        # Component View - GUI Application Detail
        component lidarSystem.guiApp "GUIComponents" {
            title "GUI Application - Internal Components"
            description "Detailed view of the GUI application showing user interface components and their interactions"
            include *
            autoLayout tb
        }

        # Component View - Vehicle Analysis Detail
        component lidarSystem.vehicleAnalyzer "VehicleAnalysisComponents" {
            title "Vehicle Analysis Engine - Processing Pipeline"
            description "Detailed view of the vehicle analysis pipeline showing data processing stages"
            include *
            autoLayout lr
        }


        # Dynamic View - Analysis Workflow
        dynamic lidarSystem "AnalysisWorkflow" {
            title "Vehicle Analysis Workflow"
            description "Step-by-step process flow for analyzing vehicle LIDAR data"

            user -> lidarSystem.launcherApp "1. Launch application"
            lidarSystem.launcherApp -> lidarSystem.guiApp "2. Start GUI mode"
            user -> lidarSystem.guiApp "3. Select E57 file"
            lidarSystem.guiApp -> lidarSystem.fileProcessor "4. Load point cloud data"
            lidarSystem.fileProcessor -> fileSystem "5. Read E57 file"
            lidarSystem.guiApp -> lidarSystem.vehicleAnalyzer "6. Start analysis"
            lidarSystem.vehicleAnalyzer -> lidarSystem.spatialAnalyzer "7. Perform spatial analysis"
            lidarSystem.vehicleAnalyzer -> lidarSystem.outputManager "8. Save results"
            lidarSystem.outputManager -> fileSystem "9. Write output files"
            lidarSystem.vehicleAnalyzer -> lidarSystem.visualizationEngine "10. Create visualization"
            lidarSystem.visualizationEngine -> lidarSystem.vtkSafetyManager "11. Setup VTK safely"
            lidarSystem.guiApp -> user "12. Display results"
        }

        # Dynamic View - Interactive Cube Selection
        dynamic lidarSystem "CubeSelectionWorkflow" {
            title "Interactive Cube Selection Workflow"
            description "Process flow for interactive cube-based region selection"

            user -> guiApp "1. Navigate to Cube Selection tab"
            user -> guiApp "2. Configure selection options"
            guiApp -> interactiveSelector "3. Launch cube selector"
            interactiveSelector -> vtkSafetyManager "4. Initialize VTK safely"
            interactiveSelector -> visualizationEngine "5. Create 3D scene"
            user -> interactiveSelector "6. Manipulate cube widget"
            interactiveSelector -> spatialAnalyzer "7. Calculate selection"
            interactiveSelector -> outputManager "8. Save selected points"
            outputManager -> fileSystem "9. Write selection file"
            interactiveSelector -> guiApp "10. Return selection results"
            guiApp -> user "11. Display selection summary"
        }

        # Filtered views for specific aspects
        container lidarSystem "ProcessingPipeline" {
            title "Core Processing Pipeline"
            description "Focus on data processing and analysis components"
            include vehicleAnalyzer fileProcessor spatialAnalyzer outputManager
            include ->vehicleAnalyzer fileProcessor-> spatialAnalyzer-> outputManager->
            autoLayout lr
        }

        container lidarSystem "UserInterface" {
            title "User Interface Architecture"
            description "Focus on user interaction and interface components"
            include launcherApp guiApp macosLauncher interactiveSelector humanPositioner
            include user->launcherApp guiApp-> interactiveSelector-> humanPositioner->
            autoLayout tb
        }

        container lidarSystem "VisualizationStack" {
            title "3D Visualization Architecture"
            description "Focus on visualization and rendering components"
            include visualizationEngine vtkSafetyManager interactiveSelector humanPositioner vtkLibrary
            include visualizationEngine->vtkSafetyManager interactiveSelector->vtkSafetyManager humanPositioner->vtkSafetyManager vtkSafetyManager->vtkLibrary
            autoLayout tb
        }

        # Styling
        styles {
            element "Person" {
                color "#ffffff"
                background "#08427b"
                shape person
            }

            element "External" {
                color "#999999"
                background "#cccccc"
            }

            element "System" {
                color "#ffffff"
                background "#1168bd"
            }

            element "Application" {
                color "#ffffff"
                background "#2e7d32"
            }

            element "GUI" {
                color "#ffffff"
                background "#7b1fa2"
            }

            element "Platform" {
                color "#ffffff"
                background "#f57c00"
            }

            element "Engine" {
                color "#ffffff"
                background "#c62828"
            }

            element "Processor" {
                color "#ffffff"
                background "#6a1b9a"
            }

            element "Renderer" {
                color "#ffffff"
                background "#00695c"
            }

            element "Tool" {
                color "#ffffff"
                background "#5d4037"
            }

            element "Analyzer" {
                color "#ffffff"
                background "#bf360c"
            }

            element "Utility" {
                color "#ffffff"
                background "#424242"
            }

            element "Support" {
                color "#ffffff"
                background "#37474f"
            }

            element "Config" {
                color "#ffffff"
                background "#455a64"
            }

            element "Storage" {
                color "#ffffff"
                background "#ff8f00"
            }

            relationship "Relationship" {
                color "#707070"
                thickness 2
            }

            relationship "Internal" {
                color "#999999"
                style dashed
            }
        }
    }
}