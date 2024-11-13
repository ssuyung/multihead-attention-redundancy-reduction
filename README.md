# cs263_final

NLP Final Project: Investigating Attention Mechanisms in Large Language Models

This project explores how different attention mechanisms affect the performance of large language models (LLMs) and examines the impact of fine-tuning on these mechanisms. We use heatmap visualizations to analyze and interpret the attention mask behavior across various experiments.

## Table of Contents
	•	Project Overview
	•	Repository Structure
	•	Getting Started
	•	Usage
	•	Experiments
	•	Results
	•	Contributing

### Project Overview

Attention mechanisms are at the core of most modern NLP models, influencing the model’s ability to learn contextual relationships within sequences. In this project, we:

	1.	Investigate different attention mechanisms and their effects on model performance.
	2.	Experiment with fine-tuning to assess changes in attention behavior.
	3.	Visualize and analyze attention heatmaps to gain insights into model focus and interpretability.

### Repository Structure
```
cs263_final/
├── README.md                 # Project introduction, goals, and structure
├── CONTRIBUTING.md           # Contribution guidelines
├── requirements.txt          # Dependencies required for running the project
├── main.py                   # Main script to run the project
├── src/                      # Source code for project-specific functions and modules
│   ├── __init__.py           # Initialize the src module
│   ├── data_processing.py    # Data loading and preprocessing functions
│   ├── model.py              # Model architecture and related functions
│   ├── evaluation.py         # Evaluation metrics and analysis functions
│   └── attention_mechanisms/ # Folder for different attention mechanism modules
│       ├── self_attention.py
│       └── cross_attention.py
├── utils/                    # Helper functions
│   ├── logging_utils.py      # Functions to log experiment parameters and metrics
│   ├── plot_utils.py         # Functions to create heatmaps and other visualizations
│   └── data_utils.py         # Data loading and processing helpers
├── experiment_results/       # Experiment results and related notebooks
│   ├── fine_tuning/
│   │   ├── experiment1_finetune.ipynb
│   │   └── experiment2_finetune.ipynb
│   ├── attention_variants/
│   │   ├── experiment1_self_attention.ipynb
│   │   └── experiment2_cross_attention.ipynb
│   └── analysis/
│       ├── heatmap_visualization.ipynb
│       └── result_comparison.ipynb
├── configs/                  # Configuration files for reproducibility
│   ├── base_config.yaml      # Common settings shared across experiments
│   ├── fine_tuning/
│   │   ├── exp1_config.yaml
│   │   └── exp2_config.yaml
│   └── attention_mechanisms/
│       ├── self_attention.yaml
│       └── cross_attention.yaml
├── data/                     # Folder for sample data and dataset setup instructions
│   ├── sample_data.csv       # A small dataset for testing
│   └── README.md             # Instructions for setting up the full dataset
├── docs/                     # Detailed project documentation
│   ├── Overview.md           # Detailed background on the project
│   ├── Methodology.md        # Explanation of attention mechanisms and experiments
│   ├── Installation.md       # Setup and installation instructions
│   └── Analysis.md           # How to interpret heatmap and other result analysis
├── scripts/                  # Scripts for setup and common commands
│   ├── setup_env.sh          # Sets up environment and installs dependencies
│   └── run_experiment.sh     # Wrapper script to run main.py with experiment configs
└── tests/                    # Unit tests for src code
    ├── test_data_processing.py
    ├── test_model.py
    └── test_evaluation.py
```
### Folder Descriptions

	•	src/: Contains source code, including data preprocessing, model definitions, and evaluation methods.
	•	experiment_results/: Includes Jupyter notebooks for experiments and heatmap analysis.
	•	configs/: Stores configuration files to ensure reproducibility across different experiments.

### Getting Started

#### Prerequisites

Ensure you have Python 3.8+ installed. You can set up the required dependencies using:

```bash
pip install -r requirements.txt
```

#### Configurations

Default configurations are provided in configs/default_config.yaml. Modify this file to customize experiment parameters.

### Usage

To run the main experiment, use:

```python main.py --config configs/default_config.yaml```

Experiment settings (e.g., attention mechanism types, dataset selection) can be adjusted in the config file. The main.py script will load these configurations automatically.

### Experiments

Each experiment is documented in the experiment_results/ folder:

	•	analysis_heatmap.ipynb: Notebook for visualizing attention heatmaps.
	•	experiment_X.ipynb: Placeholder for specific experiment setups (fine-tuning, different attention types, etc.).

### Results

Results and analysis are documented within each Jupyter notebook. The heatmap visualizations show how attention is distributed across model layers and input sequences, helping us understand how each mechanism interprets the data.

### Contributing

We welcome contributions! Please follow these steps:

	1.	Fork this repository.
	2.	Create a branch for your feature/experiment (feature/your-feature or experiment/your-experiment).
	3.	Commit your changes following the commit format: [Type] Short description.
	4.	Open a pull request for review.
