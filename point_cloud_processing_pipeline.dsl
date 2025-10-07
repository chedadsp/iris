workspace "LIDAR Processing Pipeline" "Detailed view of the LIDAR point cloud processing algorithms and data flow" {

    !identifiers hierarchical

    model {
        # Data Sources
        lidarScanner = person "LIDAR Scanner Operator" "Person operating LIDAR scanning equipment to capture vehicle data" "Operator"
        analyst = person "Data Analyst" "Researcher analyzing processed LIDAR data for insights" "Analyst"

        # Data Systems
        rawDataSystem = softwareSystem "Raw LIDAR Data" "Unprocessed point cloud data from scanning equipment" "External"
        processedDataSystem = softwareSystem "Processed Data Repository" "Analyzed and segmented point cloud results" "External"

        # Main Processing System
        processingPipeline = softwareSystem "LIDAR Processing Pipeline" "Algorithmic pipeline for vehicle point cloud analysis and interior extraction" "System" {

            # Input Layer
            dataIngestion = container "Data Ingestion Layer" "Multi-format LIDAR file reading and validation" "Python" "Input" {
                e57Reader = component "E57 Reader" "Parses E57 files using pye57 library with scan selection" "pye57" "Reader"
                pcdReader = component "PCD Reader" "ASCII PCD file parser with intensity extraction" "Custom Parser" "Reader"
                bagReader = component "ROS Bag Reader" "Extracts PointCloud2 data from ROS bag files" "bagpy" "Reader"
                formatDetector = component "Format Detector" "Automatic file format identification and validation" "Python" "Detector"
                dataValidator = component "Data Validator" "Point cloud data validation and security checks" "NumPy" "Validator"
            }

            # Core Processing Engine
            analysisEngine = container "Point Cloud Analysis Engine" "Advanced algorithms for vehicle structure identification" "Python/scikit-learn" "Engine" {
                # Preprocessing Stage
                noiseFilter = component "Noise Filter" "Statistical outlier removal and distance-based filtering" "NumPy/SciPy" "Filter"
                downsampler = component "Point Cloud Downsampler" "Adaptive sampling for performance optimization" "NumPy" "Optimizer"

                # Ground Analysis Stage
                ransacGroundFitter = component "RANSAC Ground Fitter" "Robust plane fitting for ground surface detection" "scikit-learn" "RANSAC"
                heightClassifier = component "Height-based Classifier" "Simple height threshold classification fallback" "NumPy" "Classifier"

                # Vehicle Detection Stage
                dbscanClusterer = component "DBSCAN Clusterer" "Density-based spatial clustering for vehicle identification" "scikit-learn" "DBSCAN"
                clusterAnalyzer = component "Cluster Analyzer" "Geometric analysis of clusters for vehicle selection" "NumPy" "Analyzer"

                # Interior Extraction Stage
                voxelizer = component "3D Voxelizer" "Converts point cloud to 3D occupancy grid" "NumPy" "Voxelizer"
                distanceTransform = component "Distance Transform" "3D distance field calculation for interior detection" "SciPy" "Transform"
                interiorDetector = component "Interior Detector" "Identifies passenger compartment regions" "SciPy/NumPy" "Detector"
                cockpitRefiner = component "Cockpit Refiner" "Geometric refinement using vehicle orientation analysis" "NumPy" "Refiner"

                # Density Analysis
                nearestNeighbors = component "Nearest Neighbors Analyzer" "K-NN analysis for point density assessment" "scikit-learn" "KNN"
                densityFilter = component "Density Filter" "Filters points based on local density characteristics" "NumPy" "Filter"
            }

            # Spatial Analysis System
            spatialProcessor = container "3D Spatial Processor" "Advanced spatial analysis and occupancy modeling" "Python/SciPy" "Spatial" {
                occupancyMapper = component "Occupancy Mapper" "Creates 3D occupancy grids from point clouds" "NumPy" "Mapper"
                distanceCalculator = component "Distance Calculator" "Euclidean distance transform in 3D space" "SciPy" "Calculator"
                morphologyProcessor = component "Morphology Processor" "3D morphological operations for shape analysis" "SciPy" "Morphology"
                volumeAnalyzer = component "Volume Analyzer" "Analyzes interior volumes and accessibility" "NumPy" "Volume"
            }

            # Human Model Integration
            humanModelSystem = container "Human Model System" "Anatomically accurate human model generation and positioning" "Python" "HumanModel" {
                bodyGenerator = component "Body Model Generator" "Generates anatomically proportioned human point clouds" "NumPy" "Generator"
                poseController = component "Pose Controller" "Controls human model positioning and orientation" "NumPy" "Controller"
                scaleAdjuster = component "Scale Adjuster" "Adjusts model scale for different human sizes" "NumPy" "Adjuster"
                sceneIntegrator = component "Scene Integrator" "Integrates human models with vehicle interiors" "NumPy" "Integrator"
            }

            # Output Management
            resultManager = container "Result Management System" "Processing result storage and metadata management" "Python/NumPy" "Output" {
                dataSerializer = component "Data Serializer" "Saves point cloud arrays as NumPy files" "NumPy" "Serializer"
                metadataManager = component "Metadata Manager" "Manages analysis metadata and parameters" "JSON" "Metadata"
                fileOrganizer = component "File Organizer" "Organizes output files with naming conventions" "pathlib" "Organizer"
                progressTracker = component "Progress Tracker" "Tracks and reports analysis progress" "Python" "Tracker"
            }

            # Processing Relationships - Data Flow

            # Input processing flow
            rawDataSystem -> dataIngestion "Provides raw LIDAR files"
            dataIngestion.formatDetector -> dataIngestion.e57Reader "Routes E57 files"
            dataIngestion.formatDetector -> dataIngestion.pcdReader "Routes PCD files"
            dataIngestion.formatDetector -> dataIngestion.bagReader "Routes BAG files"
            dataIngestion.dataValidator -> analysisEngine "Validates and forwards data"

            # Analysis engine internal flow
            analysisEngine.noiseFilter -> analysisEngine.downsampler "Filters outliers"
            analysisEngine.downsampler -> analysisEngine.ransacGroundFitter "Samples for ground fitting"
            analysisEngine.ransacGroundFitter -> analysisEngine.dbscanClusterer "Separates non-ground points"
            analysisEngine.heightClassifier -> analysisEngine.dbscanClusterer "Fallback ground separation"
            analysisEngine.dbscanClusterer -> analysisEngine.clusterAnalyzer "Clusters point data"
            analysisEngine.clusterAnalyzer -> analysisEngine.voxelizer "Identifies vehicle points"
            analysisEngine.voxelizer -> spatialProcessor "Creates spatial representation"

            # Spatial processing flow
            spatialProcessor.occupancyMapper -> spatialProcessor.distanceCalculator "Maps occupancy"
            spatialProcessor.distanceCalculator -> spatialProcessor.morphologyProcessor "Calculates distances"
            spatialProcessor.morphologyProcessor -> spatialProcessor.volumeAnalyzer "Processes shapes"
            spatialProcessor.volumeAnalyzer -> analysisEngine.distanceTransform "Analyzes volumes"

            # Interior detection flow
            analysisEngine.distanceTransform -> analysisEngine.interiorDetector "Transforms distances"
            analysisEngine.interiorDetector -> analysisEngine.nearestNeighbors "Detects interior regions"
            analysisEngine.nearestNeighbors -> analysisEngine.densityFilter "Analyzes point density"
            analysisEngine.densityFilter -> analysisEngine.cockpitRefiner "Filters by density"
            analysisEngine.cockpitRefiner -> resultManager "Refines cockpit detection"

            # Human model integration
            analysisEngine.cockpitRefiner -> humanModelSystem.bodyGenerator "Provides interior dimensions"
            humanModelSystem.bodyGenerator -> humanModelSystem.poseController "Generates body model"
            humanModelSystem.poseController -> humanModelSystem.scaleAdjuster "Controls pose"
            humanModelSystem.scaleAdjuster -> humanModelSystem.sceneIntegrator "Adjusts scale"
            humanModelSystem.sceneIntegrator -> resultManager "Integrates with scene"

            # Output management
            resultManager.dataSerializer -> processedDataSystem "Saves analysis results"
            resultManager.metadataManager -> processedDataSystem "Saves metadata"
            resultManager.progressTracker -> analyst "Reports progress"
        }

        # External Algorithm Libraries
        scikitLearn = softwareSystem "scikit-learn" "Machine learning library providing DBSCAN, RANSAC, and KNN algorithms" "External"
        scipyStack = softwareSystem "SciPy Stack" "Scientific computing library providing distance transforms and morphology" "External"
        numpyLibrary = softwareSystem "NumPy" "Numerical computing library for array operations and linear algebra" "External"

        # Algorithm dependencies
        processingPipeline -> scikitLearn "Uses ML algorithms"
        processingPipeline -> scipyStack "Uses spatial algorithms"
        processingPipeline -> numpyLibrary "Uses array operations"

        # User interactions
        lidarScanner -> rawDataSystem "Captures LIDAR data"
        analyst -> processingPipeline "Configures and monitors analysis"
        analyst -> processedDataSystem "Reviews results"
    }

    views {
        # System Context for Processing Pipeline
        systemContext processingPipeline "ProcessingSystemContext" {
            title "LIDAR Processing Pipeline - System Context"
            description "High-level view of the processing pipeline and its data sources and consumers"
            include *
            autoLayout lr
        }

        # Container View - Processing Architecture
        container processingPipeline "ProcessingContainers" {
            title "LIDAR Processing Pipeline - Container Architecture"
            description "Major processing containers and their relationships in the analysis pipeline"
            include *
            autoLayout tb
        }

        # Component View - Analysis Engine Detail
        component processingPipeline.analysisEngine "AnalysisEngineDetail" {
            title "Point Cloud Analysis Engine - Algorithm Components"
            description "Detailed view of the core analysis algorithms and their processing sequence"
            include *
            autoLayout lr
        }

        # Component View - Spatial Processing Detail
        component processingPipeline.spatialProcessor "SpatialProcessingDetail" {
            title "3D Spatial Processor - Spatial Analysis Components"
            description "Detailed view of spatial analysis algorithms for 3D occupancy and volume analysis"
            include *
            autoLayout tb
        }

        # Dynamic View - Complete Processing Workflow
        dynamic processingPipeline "CompleteProcessingWorkflow" {
            title "Complete LIDAR Processing Workflow"
            description "End-to-end processing flow from raw data to analyzed results"

            lidarScanner -> rawDataSystem "1. Capture LIDAR scan data"
            rawDataSystem -> processingPipeline.dataIngestion "2. Input raw point cloud files"
            processingPipeline.dataIngestion -> processingPipeline.analysisEngine "3. Validate and parse data"
            processingPipeline.analysisEngine -> processingPipeline.spatialProcessor "4. Create spatial representation"
            processingPipeline.spatialProcessor -> processingPipeline.analysisEngine "5. Return spatial analysis"
            processingPipeline.analysisEngine -> processingPipeline.humanModelSystem "6. Integrate human models"
            processingPipeline.humanModelSystem -> processingPipeline.resultManager "7. Save integrated results"
            processingPipeline.resultManager -> processedDataSystem "8. Store final analysis"
            processedDataSystem -> analyst "9. Provide analysis results"
        }


        # Filtered Views for Different Aspects

        # Data Flow Focus
        container processingPipeline "DataFlowView" {
            title "Data Flow Architecture"
            description "Focus on data movement and transformation through the processing pipeline"
            include processingPipeline.dataIngestion processingPipeline.analysisEngine processingPipeline.spatialProcessor processingPipeline.resultManager
            include processingPipeline.dataIngestion->processingPipeline.analysisEngine processingPipeline.analysisEngine->processingPipeline.spatialProcessor processingPipeline.spatialProcessor->processingPipeline.analysisEngine processingPipeline.analysisEngine->processingPipeline.resultManager
            autoLayout lr
        }

        # Algorithm Focus
        component processingPipeline.analysisEngine "AlgorithmView" {
            title "Core Analysis Algorithms"
            description "Focus on the machine learning and mathematical algorithms"
            include *
            autoLayout lr
        }

        # 3D Processing Focus
        component processingPipeline.spatialProcessor "SpatialProcessingView" {
            title "3D Spatial Processing Algorithms"
            description "Focus on spatial analysis and 3D geometric processing"
            include *
            autoLayout lr
        }

        # Styling for Processing Pipeline
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

            element "Input" {
                color "#ffffff"
                background "#2e7d32"
            }

            element "Engine" {
                color "#ffffff"
                background "#c62828"
            }

            element "Spatial" {
                color "#ffffff"
                background "#6a1b9a"
            }

            element "HumanModel" {
                color "#ffffff"
                background "#00695c"
            }

            element "Output" {
                color "#ffffff"
                background "#ff8f00"
            }

            element "Reader" {
                color "#ffffff"
                background "#388e3c"
            }

            element "Filter" {
                color "#ffffff"
                background "#5d4037"
            }

            element "RANSAC" {
                color "#ffffff"
                background "#d32f2f"
            }

            element "DBSCAN" {
                color "#ffffff"
                background "#7b1fa2"
            }

            element "Transform" {
                color "#ffffff"
                background "#f57c00"
            }

            element "Detector" {
                color "#ffffff"
                background "#303f9f"
            }

            element "KNN" {
                color "#ffffff"
                background "#689f38"
            }

            element "Generator" {
                color "#ffffff"
                background "#0097a7"
            }

            element "Serializer" {
                color "#ffffff"
                background "#fbc02d"
            }

            relationship "Relationship" {
                color "#707070"
                thickness 2
            }

            relationship "Algorithm" {
                color "#d32f2f"
                thickness 3
            }

            relationship "Data" {
                color "#1976d2"
                thickness 2
                style dashed
            }
        }
    }
}