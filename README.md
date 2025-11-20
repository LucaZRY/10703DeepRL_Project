
## Project Structure

```
10703DeepRL_Project/
├── README.md                          # This file
├── expert_data_generator.py           # Clean expert data generation
├── data_validator.py                  # Data validation and quality control
├── ppo_expert.py                      # PPO expert policy implementation
├── diffusion_dagger_trainer.py        # Main diffusion-DAgger trainer
├── environment.yml                    # Conda environment specification
├── 10703DeepRL_Project-Lucas/         
├── 10703DeepRL_Project-Tianyun/       
└── data/                              # Generated datasets
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd 10703DeepRL_Project

# Create conda environment
conda env create -f environment.yml
conda activate deeprl-diffusion

# Verify installation
python -c "import torch, gymnasium, cv2, numpy; print('✅ Dependencies ready')"
```

### 2. Train Expert Policy

```bash
# Train PPO expert for CarRacing-v2
python ppo_expert.py

# This will generate: ppo_discrete_carracing.pt
```

### 3. Generate Expert Data

```bash
# Generate standard expert demonstrations
python expert_data_generator.py --episodes 100 --output expert_data

# Generate recovery scenarios
python expert_data_generator.py --recovery --episodes 50 --output recovery_data

# Generate with videos for analysis
python expert_data_generator.py --episodes 20 --videos --output demo_data
```

### 4. Validate Data Quality

```bash
# Comprehensive validation with visualizations
python data_validator.py --input expert_data/expert_dataset.pkl --output validation_report --visualizations

# Check data integrity only
python data_validator.py --input expert_data/expert_dataset.pkl
```

### 5. Train Diffusion-DAgger Policy

```bash
# Train with offline distillation
python diffusion_dagger_trainer.py --mode offline --expert_data expert_data/expert_dataset.pkl

# Train with online adaptation
python diffusion_dagger_trainer.py --mode online --recovery_data recovery_data/
```

## Core Components

### Expert Data Generation (`expert_data_generator.py`)

**Features:**
- Clean, production-ready expert demonstration collection
- Recovery scenario generation for robust learning
- Multiple export formats (pickle, HDF5, NPZ)
- Comprehensive statistics and analysis
- Video recording capabilities

**Usage:**
```bash
python expert_data_generator.py [OPTIONS]

Options:
  --episodes INT       Number of episodes to collect (default: 100)
  --output PATH        Output directory (default: expert_data)
  --recovery          Collect recovery scenarios
  --videos            Save video recordings
  --format FORMAT     Output format: pickle, hdf5, npz (default: pickle)
```

### Data Validation (`data_validator.py`)

**Features:**
- Comprehensive data integrity checks
- Expert performance analysis
- Statistical anomaly detection
- Automated visualization generation
- Quality scoring and recommendations

**Usage:**
```bash
python data_validator.py [OPTIONS]

Options:
  --input PATH         Input dataset file
  --output PATH        Output report prefix
  --visualizations     Generate plots
  --anomaly_threshold  Z-score threshold for anomalies (default: 2.5)
```

### PPO Expert Policy (`ppo_expert.py`)

Discrete PPO implementation for CarRacing-v2 with:
- Frame stacking and preprocessing
- Discrete action space mapping
- Expert wrapper for continuous action generation

### Diffusion-DAgger Trainer (`diffusion_dagger_trainer.py`)

Main training pipeline implementing:
- Trajectory diffusion model as expert
- Offline and online distillation modes
- Recovery trajectory generation
- Performance monitoring and evaluation

## Methodology

### 1. Offline Distillation Integration (Week 3)

- Train diffusion model on expert trajectories
- Use diffusion model to generate recovery behaviors
- Distill knowledge into student policy offline

### 2. Online Distillation Integration (Week 4)

- Interactive adaptation during training
- Query diffusion expert for failure scenarios
- Online learning from synthetic recovery data

### 3. Robustness Analysis (Week 5)

- Evaluate performance under perturbations
- Compare against baseline policies (PPO, BC)
- Analyze sample efficiency improvements

## Data Formats

### Expert Demonstrations Structure

```python
{
    'demonstrations': [
        {
            'episode_id': int,
            'observations': np.ndarray,  # Shape: (T, 4, 84, 84)
            'actions': np.ndarray,       # Shape: (T, 3) - [steer, gas, brake]
            'rewards': np.ndarray,       # Shape: (T,)
            'episode_length': int,
            'total_reward': float
        }
    ],
    'statistics': {...},
    'config': {...},
    'metadata': {...}
}
```

### Recovery Data Structure

```python
{
    'recovery_episodes': [...],  # Similar to demonstrations
    'perturbation_metadata': {
        'perturbation_types': [str],
        'recovery_success_rates': dict
    }
}
```

## Performance Targets

### Expert Quality Metrics
- **Success Rate**: > 70% (episodes with return > 700)
- **Average Return**: > 600 points
- **Episode Length**: 200-1000 steps
- **Action Smoothness**: Low consecutive action variance

### Validation Checks
- **Data Integrity**: No NaN/inf values
- **Shape Consistency**: Proper tensor dimensions
- **Action Ranges**: Valid continuous control bounds
- **Temporal Consistency**: Proper episode boundaries

## Expected Results

I anticipate:
1. **Improved Sample Efficiency**: Learning from synthetic recovery data
2. **Enhanced Robustness**: Better handling of failure scenarios
3. **Training Stability**: Reduced variance through trajectory-level guidance

Success metrics:
- 20%+ improvement in sample efficiency vs. standard BC/DAgger
- 15%+ improvement in robustness to perturbations
- Stable training convergence across multiple seeds

## Implementation Notes

### CarRacing Environment
- Discrete action space: 5 actions → continuous mapping
- Frame preprocessing: 84×84 grayscale, 4-frame stack
- Success threshold: 700+ points (track completion)

### Diffusion Model Architecture
- U-Net based trajectory diffusion
- Temporal conditioning for action sequences
- Noise scheduling for generation control

### Recovery Strategy
- Perturbation injection during training
- Trajectory-level expert queries
- Multi-step recovery planning

## Troubleshooting

### Common Issues

1. **Expert Model Not Found**
   ```bash
   # Train expert first
   python ppo_expert.py
   ```

2. **Memory Issues During Collection**
   ```bash
   # Reduce episode count
   python expert_data_generator.py --episodes 50
   ```

3. **Data Validation Failures**
   ```bash
   # Check validation report for specific issues
   python data_validator.py --input dataset.pkl --visualizations
   ```

### Performance Optimization
- Use GPU acceleration when available
- Enable data compression for large datasets
- Batch data processing for efficiency




## References

1. **Chi, C., et al.** (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*.

2. **Ross, S., Gordon, G., & Bagnell, D.** (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.

3. **Lee, J., et al.** (2025). Diff-DAgger: Uncertainty-Guided Diffusion for Imitation Learning.

4. **Zhang, L., et al.** (2024). DMD: Diffusion Models for Trajectory Generation.

5. **Shi, H., et al.** (2025). DifNav: Diffusion-Based Navigation with DAgger Training.

