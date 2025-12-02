# 10703DeepRL_Project

Proposal link: https://docs.google.com/document/d/13x7v027wpC0Gban6nfV_T_44joJ1H9XwViIlBKYY_ew/edit?usp=sharing


PPO.py  
    → carracing_ppo_dataset_fast.npz
        ↓
convert_ppo_expert.py  
    → data/expert_carracing/(states.npy, actions.npy)
        ↓
train_diffusion.py  
    → diffusion checkpoint (expert.pt)

    python train_diffusion.py \
  --expert_dir data/expert_carracing \
  --save_path results/diffusion_expert/carracing_expert_debug.pt \
  --train_steps 2000 \
  --batch_size 256

        ↓
generate_synthetic_carracing.py  
    → data/generated_carracing/(states.npy, actions.npy)

    python generate_synthetic_carracing.py \
  --expert_dir data/expert_carracing \
  --ckpt_path results/diffusion_expert/carracing_expert.pt \
  --out_dir data/generated_carracing \
  --num_samples 200000


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


