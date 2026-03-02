# KalmanNet

## Feb.13, 2023 Update "batched"

Support a batch of sequences being processed simultaneously, leading to dramatic efficiency improvement.

## End-to-End Latent VIO Extension

This repository extends the original KalmanNet with an **End-to-End Latent Visual-Inertial Odometry (VIO)** system. The VIO system replaces the Euclidean state-space model with a manifold-aware architecture built on SE(3) Lie groups using [PyPose](https://pypose.org/).

### Architecture Overview

```
Image Pair (I_{t-1}, I_t)         IMU Data (acc, gyro)
        |                                |
  SpatiotemporalEncoder           IMU Preintegration
  (ResNet18/50 backbone)          (PyPose SO3/SE3)
        |                                |
    z_t (latent obs)            x̂_{t|t-1} (prior state)
        |                                |
        |    LatentObservationModel       |
        |    ẑ_t = h_φ(x̂_{t|t-1})       |
        |              |                  |
        +--- Δz_t = z_t - ẑ_t -----------+
                       |
              ManifoldKalmanNet
              (GRU_Q, GRU_Σ, GRU_S)
                       |
                   K_t (Kalman Gain)
                       |
        x̂_{t|t} = x̂_{t|t-1} ⊞ (K_t · Δz_t)
              (SE3 Exp map update)
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| Visual Encoder | `VIO/__init__.py` | ResNet18/50 backbone processing consecutive image pairs into latent observation vectors |
| Observation Model | `VIO/latent_observation_model.py` | Learnable MLP mapping physical state to visual latent space |
| VIO System Model | `VIO/vio_system_model.py` | SE(3) state model with PyPose IMU preintegration |
| Manifold KalmanNet | `KNet/ManifoldKalmanNet_nn.py` | GRU-based Kalman gain estimator using Lie algebra differences |
| Training Pipeline | `Pipelines/Pipeline_KF_visual.py` | 3-stage curriculum training with geodesic loss |
| Main Entry Point | `main_VIO.py` | End-to-end training, testing, and demo |

### Mathematical Foundations

**State Representation**: The state vector lives on the SE(3) manifold via PyPose `LieTensor`:
- Pose: `pp.SE3` (translation + unit quaternion)
- Velocity: `R^3`

**Prior Prediction**: Uses IMU preintegration on SO(3):
```
x̂_{t|t-1} = f_IMU(x_{t-1|t-1}, acc_t, gyro_t)
```

**Innovation**: Computed in the learned latent space:
```
Δz_t = z_t - h_φ(x̂_{t|t-1})
```

**Posterior Update**: Uses exponential map (box-plus) on SE(3):
```
x̂_{t|t} = x̂_{t|t-1} ⊞ (K_t · Δz_t)
```

**Loss Function**: Geodesic loss on SE(3) instead of Euclidean MSE:
```
L = α · L_geo(pose_pred, pose_gt) + (1-α) · L_vel(v_pred, v_gt)
```

### Three-Stage Curriculum Training

| Stage | Trainable Modules | Frozen Modules | Purpose |
|-------|-------------------|----------------|---------|
| Stage 1 | Encoder + ObsModel | KalmanNet | Pre-train visual frontend and latent mapping |
| Stage 2 | KalmanNet | Encoder + ObsModel | Train filtering GRUs with ground truth guidance |
| Stage 3 | All | None | Joint fine-tuning via BPTT |

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 1.10.1`
- `torchvision >= 0.11.2`
- `pypose >= 0.7.0`
- `timm >= 0.9.0`

## Running the VIO System

### Quick Demo (Synthetic Data)

```bash
python main_VIO.py --mode demo --use_cuda False --pretrained False
```

### Training

```bash
python main_VIO.py --mode train \
    --backbone resnet18 \
    --latent_dim 128 \
    --stage1_steps 200 \
    --stage2_steps 200 \
    --stage3_steps 100 \
    --stage1_lr 1e-4 \
    --stage2_lr 1e-4 \
    --stage3_lr 1e-5 \
    --n_batch 4 \
    --save_dir results/vio
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `resnet18` | Visual backbone (`resnet18` or `resnet50`) |
| `--latent_dim` | `128` | Dimension of latent observation space |
| `--state_dim` | `9` | State vector dimension (3 trans + 3 rot + 3 vel) |
| `--stage1_steps` | `200` | Number of Stage 1 training steps |
| `--stage2_steps` | `200` | Number of Stage 2 training steps |
| `--stage3_steps` | `100` | Number of Stage 3 training steps |
| `--n_batch` | `4` | Batch size (number of sequences per step) |
| `--use_cuda` | `True` | Use GPU acceleration |
| `--save_dir` | `results/vio` | Directory to save checkpoints |

### Testing

```bash
python main_VIO.py --mode test \
    --checkpoint results/vio/model_latent_vio.pt \
    --use_cuda False
```

### Using Custom Datasets

Replace the `generate_synthetic_data` function in `main_VIO.py` with your data loader. The expected format for each sequence:

- **images**: `[T+1, 3, H, W]` tensor of RGB images
- **poses**: `[T+1, 7]` tensor of SE(3) poses (tx, ty, tz, qw, qx, qy, qz)
- **velocities**: `[T+1, 3]` tensor of linear velocities
- **accelerometer**: `[T, 3]` tensor of IMU accelerometer readings
- **gyroscope**: `[T, 3]` tensor of IMU gyroscope readings

---

## Original KalmanNet

### Link to paper

[KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043)

### Running Original Code

* Linear case (canonical model or constant acceleration model)

```bash
python3 main_linear_canonical.py
python3 main_linear_CA.py
```

* Non-linear Lorenz Attractor case (Discrete-Time, decimation, or Non-linear observation function)

```bash
python3 main_lor_DT.py
python3 main_lor_decimation.py
python3 main_lor_DT_NLobs.py
```

### Parameter settings

* `Simulations/model_name/parameters.py` — Model settings: m, n, f/F, h/H, Q and R.
* `Simulations/config.py` — Dataset size, training parameters and network settings.
* Main files — Set flags, paths, etc.
