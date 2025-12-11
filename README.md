# 10703DeepRL_Project

Proposal link: https://docs.google.com/document/d/13x7v027wpC0Gban6nfV_T_44joJ1H9XwViIlBKYY_ew/edit?usp=sharing


# CarRacing-v2 DAgger with Diffusion Policy

This project implements a **DAgger (Dataset Aggregation)** pipeline to train a robust autonomous driving agent for the `CarRacing-v2` environment. 

It uses a **Diffusion Model** as the "Teacher" (Expert) and a **CNN Policy** as the "Student." The pipeline leverages a pre-trained PPO agent to generate perfect initial data, ensuring the teacher is highly capable.

---

The workflow consists of five distinct phases:

1.  **Phase 1: Data Generation** (Create "Perfect" Expert Data using PPO)
2.  **Phase 2: Teacher Education** (Train the Diffusion Expert)
3.  **Phase 3: Student Pre-Training** (Offline Distillation)
4.  **Phase 4: DAgger Training** (Online Fine-Tuning)
5.  **Phase 5: Visualization** (Record & Plot Results)

---

Ensure you have the necessary libraries installed:

```bash
pip install gymnasium[box2d] torch numpy stable-baselines3 huggingface_sb3 shimmy imageio moviepy matplotlib diffusers

###

Phase 1: Data Generation (The "Textbook")
Since human keyboard data is often "jerky" or "lazy" (coasting), we use a pre-trained PPO Agent (Continuous Control) to generate high-quality driving data.


###

Phase 2: Teacher Education (Training the Expert)
We train a Diffusion Policy to clone the behavior of the PPO agent. This model serves as the "Teacher" during the DAgger process.

train_diffusion.py  
    → diffusion checkpoint (expert.pt)

    python train_diffusion.py \
    --expert_dir data/human_expert \
    --save_path results/diffusion_expert/perfect_expert_96.pt \
    --train_steps 50000 \
    --batch_size 64


        ↓
Crucial: Before proceeding, verify the expert drives well on new tracks.

    python test_expert.py
    Target: Score > 800.
    Fix: If score is low (< 400), generate more data (Phase 1) or train longer.


###
Phase 3: Student Pre-Training (Offline)
We pre-train the student (CNN Policy) on the static expert dataset. This prevents the student from starting "brain dead" (random) during the online DAgger phase, which speeds up convergence significantly.

train_student.py  
    → student_offline_model.pt  (BC on synthetic data)

    python train_student.py --mode offline_distill \
    --data_dir data/human_expert \
    --save_path results/student_pretrain.pt

Input: data/human_expert/expert_trajectories.npz

Output: results/student_pretrain.pt

Method: Behavior Cloning (BC) with automatic flattening of episode data.

###

Phase 4: DAgger (Online Training)
This is the core of the project. The student drives in the environment. When it drifts or struggles, the Diffusion Expert provides the correct action ("Labeling"), which is aggregated into the dataset for retraining.

Script: dagger_online.py

1. Configuration
Open dagger_online.py and ensure the paths in DAggerConfig match your generated files:

diffusion_model_path: "results/diffusion_expert/perfect_expert_96.pt"

pretrained_student_path: "results/student_pretrain.pt"

ppo_npz_path: "data/human_expert/expert_trajectories.npz"


###
Phase 5: Visualization
1. Record Video
Generate an MP4 video of your final agent driving.

python record_student.py
Output: dagger_student.mp4

2. Plot Performance
Run a fresh evaluation (20 episodes) and plot the score distribution.

python plot_results.py
Output: results/final_performance.png



.
├── data/
│   └── human_expert/
│       └── expert_trajectories.npz    # The "Textbook" (100 episodes)
├── results/
│   ├── diffusion_expert/
│   │   └── perfect_expert_96.pt       # The Teacher
│   ├── student_pretrain.pt            # The Student (Phase 3)
│   ├── student_dagger_final.pt        # The Master (Phase 4)
│   └── videos/                        # Recorded runs
├── src/
│   └── models.py                      # CNN & Diffusion Architectures
├── generate_perfect_data.py           # Phase 1: PPO Data Gen
├── train_diffusion.py                 # Phase 2: Teacher Training
├── test_expert.py                     # Phase 2: Verification
├── train_student.py                   # Phase 3: Student Training
├── dagger_online.py                   # Phase 4: DAgger Loop
├── record_student.py                  # Visualization
└── README.md