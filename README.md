# Rediscovering Orbital Mechanics with AI: A PyTorch Replication

![Project Status](https://img.shields.io/badge/status-complete-green)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/pytorch-2.0+-orange.svg)

This repository contains a PyTorch implementation and replication of the core concepts from the paper **["Rediscovering orbital mechanics with machine learning"](https://arxiv.org/abs/2202.02306)** by Lemos, Jeffrey, Cranmer et al.

The project demonstrates a complete pipeline for scientific discovery using machine learning. It starts by simulating a multi-planet system, then trains a physics-informed Graph Neural Network (GNN) to learn the system's dynamics and infer the hidden masses of the planets. Finally, it uses PySR (Symbolic Regression) to analyze the trained GNN and automatically discover the underlying analytical formula, successfully rediscovering Newton's law of universal gravitation.

## Key Features

- **N-Body Simulator:** A simple but physically accurate simulator built with NumPy to generate ground-truth training data.
- **Physics-Informed GNN:** A Graph Neural Network built with PyTorch that incorporates physical priors (like the proportional dependence on mass, F ∝ m₁*m₂) directly into its architecture.
- **Hidden Parameter Inference:** The model successfully learns the masses of the planets—parameters that were hidden from it—by observing only their motion.
- **Symbolic Law Discovery:** The trained model is distilled into a simple, human-interpretable formula using PySR, revealing the inverse-square law of gravity.
- **End-to-End Workflow:** The repository provides separate, clean scripts for training the model (`train_model.py`) and for performing the final symbolic deduction (`deduction.py`).

## File Structure

The project is organized into several modules for clarity and maintainability:

```
.
├── train_model.py          # Main script to run the simulation and train the GNN.
├── deduction.py            # Script to load the trained model and run PySR.
├── model.py                # Contains the PhysicsInteractionNetwork (GNN) class definition.
├── body.py                 # Defines the Body class for the physics simulation.
├── body_manager.py         # Manages the collection of Body objects.
├── transforms.py           # Utility functions for coordinate transformations.
├── best_model.pth          # (Generated) The saved state of the best trained GNN model.
├── best_masses.pth         # (Generated) The saved tensor of the best inferred planetary masses.
├── outputs/                # (Generated) Directory for outputs from PySR.
└── images/                 # (Generated) Directory for output plots.
    ├── imagen1_comparacion_orbitas.png
    ├── imagen2_convergencia.png
    └── imagen3_perdida.png
```

## Setup and Installation

This project requires Python 3.9+ and several scientific computing libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/isaigm/rediscovering-orbital-mechanics-with-ml
    cd rediscovering-orbital-mechanics-with-ml
    ```

2.  **Install dependencies:**
    It is highly recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install numpy matplotlib torch torch_scatter torch_geometric pysr pandas
    ```
    **Note:** The PyTorch-related libraries (`torch`, `torch_scatter`, `torch_geometric`) can have specific installation requirements depending on your CUDA version. Please refer to the official PyTorch Geometric installation instructions for the correct commands for your system. PySR will handle the installation of its Julia backend on first use.

## How to Run

The project is divided into two main steps.

### Step 1: Train the GNN Model

This script will run the physics simulation, train the GNN, save the best model, and generate plots visualizing the training process and results.

```bash
python train_model.py
```
This process is computationally intensive and will be significantly faster on a machine with a CUDA-enabled GPU. Upon completion, it will save the trained model (`best_model.pth`), the inferred masses (`best_masses.pth`), and several plots in the `images/` directory.

### Step 2: Discover the Symbolic Formula

After the model has been trained, run this script to perform symbolic regression on the knowledge learned by the GNN.

```bash
python deduction.py
```
This script will load the saved model and use PySR to find the analytical formula that best describes the learned interaction. The process can take several minutes. The final discovered formula will be printed to the console and saved in the `outputs/` directory.

## Sample Results

After a successful run, the model learns the planetary masses with high precision:

-   **Real Planet Masses:** `[3.00e-06, 9.50e-04, 3.30e-07]`
-   **Best Learned Masses:** `[3.02e-06, 9.43e-04, 4.64e-07]` (Error <1% for dominant bodies)

The `deduction.py` script then discovers the underlying physical law. A typical successful run yields a formula equivalent to:

```
F = 37.13 * (m1 * m2) / r^2
```
This result successfully rediscovers Newton's law of universal gravitation, where the constant `37.13` is a close approximation of the theoretical value of `G ≈ 39.47` in the simulation's unit system (a ~5.9% error).

## Acknowledgments

This project was inspired by the work of the authors of "Rediscovering orbital mechanics with machine learning." The code and methodology were developed through an iterative process of experimentation and debugging, with significant assistance from the AI model Gemini 2.5 Pro.
