# F1TENTH Dataset Generation Pipeline

This document describes the F1TENTH dataset generation pipeline adapted from the original CarRacing diffusion-based policy distillation project.

## Overview

The F1TENTH pipeline generates synthetic racing datasets using the same diffusion-based approach as the original CarRacing project, but adapted for F1TENTH autonomous racing with LiDAR observations and different action spaces.

## Pipeline Architecture

```
F1TENTH Environment (LiDAR + Racing Dynamics)
    ↓ (PPO Expert Training)
F1TENTH PPO Dataset (.npz)
    ↓ (Trajectory Conversion)
Expert Trajectory Dataset
    ↓ (Diffusion Training)
F1TENTH Diffusion Model
    ↓ (Synthetic Generation)
Generated Synthetic F1TENTH Dataset
    ↓ (Student Training)
F1TENTH Student Policy
```

## Key Differences from CarRacing

### Environment
- **Observations**: LiDAR scans instead of camera images
- **Action Space**: [steering_angle, speed] instead of [steer, gas, brake]
- **Dynamics**: F1TENTH vehicle model with realistic racing physics
- **Tracks**: Real F1 circuits (Spielberg, Monaco, Silverstone, etc.)

### Data Format
- **States**: (N, lidar_dim) where lidar_dim is typically 1080 points
- **Actions**: (N, 2) for [steering_angle, speed]
- **Normalization**:
  - LiDAR ranges normalized to [0, 1]
  - Steering: [-0.4189, 0.4189] radians (±24 degrees)
  - Speed: [0, 8.0] m/s

## Installation

### Prerequisites

```bash
# Create conda environment from existing environment.yml
conda env create -f environment.yml
conda activate drl-diffdist

# Install F1TENTH gym (optional, falls back to CarRacing)
pip install f1tenth-gym

# Install Box2D for CarRacing fallback
pip install swig
pip install "gymnasium[box2d]"
```

### Alternative: Install F1TENTH Requirements Only

```bash
pip install torch torchvision torchaudio
pip install gymnasium numpy matplotlib
pip install diffusers einops
pip install f1tenth-gym  # Optional
```

## Usage Pipeline

### 1. Train F1TENTH PPO Expert

```bash
# Train expert policy on F1TENTH environment
python f1tenth_ppo.py

# Output: f1tenth_ppo_dataset.npz
#   obs: (N, lidar_dim)    LiDAR observations
#   actions: (N, 2)        [steering, speed]
#   rewards: (N,)
#   dones: (N,)
```

**Configuration Options:**
- Modify `F1TenthPPOConfig` in `f1tenth_ppo.py`
- Change training steps, batch size, learning rates
- Select different F1TENTH maps in environment creation

### 2. Convert to Trajectory Format

```bash
# Convert PPO dataset to trajectory format
python convert_f1tenth_expert.py \
    --input f1tenth_ppo_dataset.npz \
    --output data/expert_f1tenth \
    --min_length 50 \
    --max_length 1000

# Output: data/expert_f1tenth/
#   states.npy:  (num_traj, T, lidar_dim)
#   actions.npy: (num_traj, T, 2)
```

**Parameters:**
- `--min_length`: Minimum trajectory length to keep
- `--max_length`: Maximum trajectory length (truncate longer)
- `--target_length`: Fixed padding length (optional)

### 3. Train Diffusion Model

```bash
# Train diffusion expert on F1TENTH trajectories
python train_diffusion_f1tenth.py \
    --expert_dir data/expert_f1tenth \
    --save_path f1tenth_diffusion_expert.pt \
    --num_layers 6 \
    --hidden_size 128 \
    --num_heads 8 \
    --num_diffusion_steps 30 \
    --train_steps 20000 \
    --batch_size 256 \
    --action_horizon 8 \
    --observation_horizon 16

# Output: f1tenth_diffusion_expert.pt
```

**Key Parameters:**
- `--action_horizon`: Number of future actions to predict
- `--observation_horizon`: Number of past LiDAR scans to use
- `--num_diffusion_steps`: Diffusion timesteps (30 for speed)

### 4. Generate Synthetic Data

```bash
# Generate synthetic F1TENTH racing data
python generate_synthetic_f1tenth.py \
    --model_path f1tenth_diffusion_expert.pt \
    --output_dir data/generated_f1tenth \
    --num_samples 50000 \
    --batch_size 256

# Output: data/generated_f1tenth/
#   states.npy:  (50000, lidar_dim)
#   actions.npy: (50000, 2)
```

### 5. Train Student Policy (Future)

The synthetic data can now be used for student policy training via:
- Behavioral Cloning (BC)
- DAgger with offline expert
- Policy distillation methods

## File Structure

```
├── f1tenth_ppo.py                    # PPO expert training
├── f1tenth_wrappers.py              # Environment wrappers
├── convert_f1tenth_expert.py        # Data conversion
├── train_diffusion_f1tenth.py       # Diffusion training
├── generate_synthetic_f1tenth.py    # Synthetic data generation
├── test_f1tenth_setup.py            # Environment testing
├── src/models.py                     # Diffusion transformer model
├── config/experiment.yaml           # Configuration
└── data/
    ├── expert_f1tenth/              # Expert trajectories
    └── generated_f1tenth/           # Synthetic data
```

## Environment Details

### F1TENTH Wrapper Features

1. **LiDAR Processing** (`F1TenthLidarWrapper`):
   - Normalizes LiDAR ranges
   - Handles infinite values
   - Optional downsampling for efficiency
   - Adds velocity information

2. **Action Standardization** (`F1TenthActionWrapper`):
   - Converts to normalized [steering, speed] format
   - Applies safety constraints
   - Handles F1TENTH multi-agent format

3. **Reward Shaping** (`F1TenthRewardWrapper`):
   - Progress-based rewards
   - Speed bonuses
   - Collision penalties
   - Smoothness rewards

### Available F1 Tracks

- Spielberg (Austria)
- Monaco
- Silverstone (UK)
- Spa (Belgium)
- Monza (Italy)
- Suzuka (Japan)
- LVMS (Las Vegas)
- Nurburgring (Germany)

## Testing

```bash
# Test environment setup and basic functionality
python test_f1tenth_setup.py
```

This validates:
- Package installation
- Environment creation
- Policy networks
- Dataset compatibility

## Troubleshooting

### F1TENTH Gym Not Available
- Pipeline automatically falls back to CarRacing-v3
- Install with: `pip install f1tenth-gym`

### Box2D Missing (CarRacing fallback)
- Install with: `pip install swig && pip install "gymnasium[box2d]"`

### GPU Training
- Automatically detects CUDA availability
- Force CPU with: `--device cpu`

### Memory Issues
- Reduce batch sizes in training scripts
- Use gradient accumulation for larger effective batch sizes

## Adapting to Other Environments

The pipeline can be adapted to other racing environments by:

1. **Environment Wrapper**: Modify `f1tenth_wrappers.py`
   - Change observation preprocessing
   - Adapt action space conversion
   - Adjust reward shaping

2. **Model Architecture**: Update `src/models.py`
   - Change input dimensions
   - Modify transformer architecture

3. **Training Scripts**: Adapt hyperparameters
   - Adjust learning rates
   - Change trajectory horizons
   - Modify diffusion steps

## Performance Tips

1. **Data Quality**: Filter poor trajectories during conversion
2. **Model Size**: Balance model capacity with training speed
3. **Diffusion Steps**: Use fewer steps (10-30) for faster generation
4. **Batch Size**: Maximize GPU utilization without OOM
5. **Caching**: Precompute and cache trajectory segments

## Citation

Based on the original CarRacing diffusion pipeline. Adapted for F1TENTH autonomous racing with LiDAR observations and realistic vehicle dynamics.

For F1TENTH environment:
- [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym)
- [F1TENTH Community](https://f1tenth.org/)

## Future Extensions

- Multi-agent F1TENTH racing
- Real-world F1TENTH deployment
- Integration with hardware platforms
- Advanced reward shaping for racing lines
- Online fine-tuning with real data