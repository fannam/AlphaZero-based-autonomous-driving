# AlphaZero-based Autonomous Driving

This repository implements an AlphaZero-inspired framework for autonomous driving using the [highway-env](https://github.com/eleurent/highway-env) environment. It leverages occupancy grids for state representation, providing an effective way to model the dynamics of autonomous driving scenarios. The repository includes multiple Jupyter notebooks for data generation, self-play, and training.

---

## Contents

### 1. `AlphaZero_highway_GPU.ipynb`
This notebook implements the AlphaZero algorithm using the occupancy grid provided by `highway-env`. The occupancy grid is a 3D tensor of shape `(21, 5)` with the following channel specifications:

- **Channels 1 to T**: Occupancy grids for the time steps `t, t-1, ..., t-T+1`.
- **Channel T+1**: Road representation.
- **Channel T+2 and T+3**: Relative velocity of the ego-vehicle to its maximum and minimum allowed speeds.
- **Channel T+4**: Absolute speed of the ego-vehicle.

This file provides a full workflow from self-play to model training. The trained model should be placed in the `AlphaZero` folder for further usage.

---

### 2. `alphazero_data_final.ipynb`
This notebook introduces a custom occupancy grid, derived from the kinematics observation provided by `highway-env`. The custom grid has the following channel specifications:

- **Channels 1 to T**: Occupancy grids for the time steps `t, t-1, ..., t-T+1`.  
  - The ego-vehicle is represented by `1` where it occupies.  
  - Other vehicles and obstacles are represented by `2` where they occupy.
- **Channel T+1**: Velocity-x of the ego and other vehicles.
- **Channel T+2**: Velocity-y of the ego and other vehicles.

Data generation for approximately 2,500 samples was performed using an NVIDIA T4 GPU, taking around 10 hours.

---

### 3. `alphazero_training_custom_cnn.ipynb`
This notebook utilizes the data generated in `alphazero_data_final.ipynb` for training. The model employs a custom convolutional neural network (CNN) and is optimized using the following loss functions:

- **Policy Loss**: Kullback–Leibler (KL) Divergence.
- **Value Loss**: Mean Squared Error (MSE).

Training is configured for 100 epochs and may take approximately 10 hours on a GPU.

---

### 4. `AlphaZero` Folder
This folder contain all code from `AlphaZero_highway_GPU.ipynb` and `evaluate.py` to experiment and evaluate models' performance using the occupancy grid provided by highway-env

### 5. `AlphaZero-custom-occupancy-grid`
Same as `AlphaZero` but use for our custom occupancy grid

## Requirements
- Python 3.8 or higher
- `highway-env` library
- NVIDIA GPU (T4 or better recommended)
- Libraries: `numpy`, `pytorch`, `matplotlib`
- See more: `requirements.txt`

Install the required libraries using:
```bash
pip install -r requirements.txt

