# Expert Data Preparation Guide
## Deep Reinforcement Learning - CarRacing Environment

### Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Expert Policy Architecture](#expert-policy-architecture)
4. [Data Collection Pipeline](#data-collection-pipeline)
5. [Data Preprocessing](#data-preprocessing)
6. [Quality Control & Validation](#quality-control--validation)
7. [Storage & Persistence](#storage--persistence)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Expert data preparation is crucial for imitation learning and DAgger algorithms. This guide provides a comprehensive approach to collecting, processing, and validating expert demonstrations for the CarRacing-v2 environment.

### Key Components
- **Expert Policy**: Pre-trained PPO agent with discrete→continuous action conversion
- **Data Collection**: Systematic collection of state-action pairs
- **Preprocessing**: Frame stacking, normalization, and action scaling
- **Quality Control**: Performance validation and data integrity checks

### Data Flow
```
Expert Policy → Environment Interaction → Raw Observations → Preprocessing → Dataset Storage
```

## Environment Setup

### Prerequisites
```bash
# Activate environment
conda activate drl-diffdist

# Verify dependencies
python -c "import gymnasium, torch, cv2, numpy; print('Dependencies OK')"
```

### Environment Configuration
- **Observation Space**: RGB images (96,96,3) → Grayscale (84,84,1) → Stacked (4,84,84)
- **Action Space**: Continuous [steer, gas, brake] where steer∈[-1,1], gas/brake∈[0,1]
- **Frame Processing**: Grayscale conversion, resizing, normalization to [0,1]

## Expert Policy Architecture

### PPO Expert Model
- **Input**: Stacked frames (4,84,84)
- **Architecture**: CNN feature extractor + policy/value heads
- **Output**: Discrete actions converted to continuous control
- **Performance**: Trained to achieve consistent lap completion

### Action Mapping
```python
# Discrete actions (PPO) → Continuous actions (Environment)
discrete_actions = [
    [-1, 0, 0],    # Turn left
    [1, 0, 0],     # Turn right
    [0, 1, 0],     # Accelerate
    [0, 0, 1],     # Brake
    [0, 0, 0],     # No-op
    # ... additional combinations
]
```

## Data Collection Pipeline

### Collection Strategies

#### 1. Pure Expert Demonstrations
- Expert controls environment completely
- Optimal trajectories for behavioral cloning
- Baseline performance measurement

#### 2. DAgger Collection
- Student attempts action
- Expert provides correct label
- Addresses distribution shift problem

#### 3. Recovery-Focused Collection
- Trigger expert during failure states
- Critical for robust policy learning
- Off-track, collision, or low-speed scenarios

### Data Volume Recommendations
- **Initial Dataset**: 50-100 expert episodes
- **DAgger Iterations**: 20-30 episodes per iteration
- **Recovery Data**: 10-20% of total dataset
- **Total Target**: 1000+ state-action pairs

## Data Preprocessing

### Observation Processing
1. **Resize**: 96×96 → 84×84 (computational efficiency)
2. **Grayscale**: RGB → single channel (reduces dimensionality)
3. **Normalization**: Pixel values [0,255] → [0,1]
4. **Frame Stacking**: Stack 4 consecutive frames for temporal information
5. **Channel Reordering**: (84,84,4) → (4,84,84) for CNN input

### Action Processing
1. **Validation**: Ensure actions within valid ranges
2. **Clipping**: Constrain to action space bounds
3. **Normalization**: Optional rescaling for training stability

## Quality Control & Validation

### Expert Performance Metrics
- **Episode Return**: Average reward per episode (target: >700)
- **Completion Rate**: Percentage of successful laps
- **Lap Time**: Consistency in completion time
- **Track Coverage**: Exploration of different track sections

### Data Integrity Checks
- **Action Range Validation**: All actions within bounds
- **Observation Shape**: Consistent tensor dimensions
- **Missing Data**: No NaN or infinite values
- **Temporal Consistency**: Smooth action transitions

### Visualization Tools
- Episode return distributions
- Action histograms
- Trajectory visualizations
- Frame preprocessing examples

## Storage & Persistence

### Data Format
```python
# Dataset structure
{
    'observations': np.array,  # Shape: (N, 4, 84, 84)
    'actions': np.array,      # Shape: (N, 3)
    'metadata': {
        'episode_returns': List[float],
        'collection_method': str,
        'expert_model': str,
        'timestamp': str
    }
}
```

### Storage Options
1. **NumPy Arrays**: Efficient for large datasets
2. **Pickle Files**: Python object serialization
3. **HDF5**: Hierarchical data format for complex structures
4. **PyTorch Tensors**: Direct integration with training

### Versioning & Backup
- Use timestamp-based naming
- Store collection parameters
- Maintain dataset statistics
- Regular backup of large datasets

## Usage Examples

### Basic Expert Collection
```python
# Load expert and collect data
expert = PPOExpertPolicy("ppo_discrete_carracing.pt")
dataset = ImitationDataset()
collect_pure_expert_data(env, expert, dataset, num_episodes=50)
```

### DAgger Iteration
```python
# Iterative improvement
for iteration in range(5):
    collect_dagger_data(env, student, expert, dataset, num_episodes=20)
    train_student_policy(student, dataset)
```

### Recovery Data Collection
```python
# Focus on challenging scenarios
dataset_recovery = collect_recovery_data(
    env, expert,
    trigger_conditions=['off_track', 'low_speed', 'collision']
)
```

## Troubleshooting

### Common Issues

#### Low Expert Performance
- **Solution**: Retrain PPO expert with longer training
- **Check**: Model loading, environment compatibility

#### Action Range Violations
- **Solution**: Add action clipping and validation
- **Check**: Discrete→continuous conversion logic

#### Memory Issues
- **Solution**: Batch data collection, streaming storage
- **Check**: Dataset size, system RAM

#### Inconsistent Preprocessing
- **Solution**: Standardize preprocessing pipeline
- **Check**: Frame stacking, normalization order

### Performance Optimization
- Use GPU for preprocessing when available
- Batch environment steps
- Implement data loading pipelines
- Monitor memory usage during collection

### Debugging Tools
- Visualization of collected trajectories
- Action distribution analysis
- Performance comparison plots
- Data pipeline validation scripts

---

## Conclusion

Proper expert data preparation is fundamental to successful imitation learning. This guide provides the foundation for collecting high-quality demonstration data that enables robust policy learning in the CarRacing environment.

### Key Takeaways
1. **Quality over Quantity**: Focus on diverse, high-performing trajectories
2. **Systematic Collection**: Use structured approaches for data gathering
3. **Validation is Critical**: Always verify data quality before training
4. **Iterative Improvement**: Use DAgger for addressing distribution shift
5. **Documentation**: Maintain clear records of data collection procedures

For implementation details, refer to the accompanying Python scripts and configuration files.