# Overall Project Architecture
The project follows a modular architecture designed for the Topside Ionosphere-Plasmasphere TEC Reconstruction. It leverages hybrid physics-informed neural networks.

## Module Relationships
- **models**: Contains the core neural network architectures (HybridModel, PhysicsConstraintEngine, etc.).
- **training**: Contains the training loops, optimizers, and loss managers.
- **evaluation**: Contains scripts and metrics for model evaluation and visualization.
- **data**: Responsible for datasets, loading, and real-data fetching.
- **utils**: Contains common helpers such as logging, seeding, and tensor operations.

## Folder Responsibilities
- `app/`: Web or UI application code for predictions.
- `configs/`: YAML configuration files.
- `colab/`: Jupyter notebooks and scripts specifically optimized for Google Colab.
- `docs/`: Project documentation.
- `notebooks/`: Exploratory data analysis and experimental notebooks.
- `validation/`: Test suites and validation frameworks.

## Data Flow
Raw Data -> Preprocessing -> Dataset -> DataLoader -> Training/Evaluation -> Predictions -> Postprocessing

## Training Flow
Initialize Model -> Load Checkpoint (optional) -> Iterate Epochs -> Forward Pass (Physics-Informed) -> Calculate Losses -> Backward Pass -> Optimizer Step -> Save Checkpoint

## Prediction Flow
Load Model Checkpoint -> Ingest Real-time Data -> Forward Pass -> Output Predictions -> Export/Visualize

