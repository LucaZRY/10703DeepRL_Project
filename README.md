# 10703DeepRL_Project

Proposal link: https://docs.google.com/document/d/13x7v027wpC0Gban6nfV_T_44joJ1H9XwViIlBKYY_ew/edit?usp=sharing


PPO.py  
    → carracing_ppo_dataset_fast.npz

    python dataset_check.py --path carracing_ppo_dataset.npz

    expected data:
    obs: (N, 4, 96, 96)
    state_dim: 36864

        ↓
convert_ppo_expert.py  
    → data/expert_carracing/(states.npy, actions.npy)

    python dataset_check.py --path data/expert_carracing

    states:  (num_traj, T, 36864)
    actions: (num_traj, T, 3)


        ↓
train_diffusion.py  
    → diffusion checkpoint (expert.pt)

    python train_diffusion.py \
  --expert_dir data/expert_carracing \
  --save_path results/diffusion_expert/carracing_expert_96.pt \
  --num_layers 6 \
  --hidden_size 128 \
  --num_heads 1 \
  --num_diffusion_steps 30 \
  --train_steps 20000 \
  --batch_size 256


        ↓
generate_synthetic_carracing.py  
    → data/generated_carracing/(states.npy, actions.npy)

    python generate_synthetic_carracing.py

    python dataset_check.py --path data/generated_carracing

    states:  (200000, 36864)
    actions: (200000, 3)


        ↓
train_student.py  
    → student_offline_model.pt  (BC on synthetic data)

    python train_student.py \
  --mode offline_distill \
  --offline_data_dir data/generated_carracing \
  --results_dir results



        ↓
OfflineExpert(PPO + Diffusion NN labeler)
        ↓
dagger_carracing_2.py  
    → DAgger training → final student policy


