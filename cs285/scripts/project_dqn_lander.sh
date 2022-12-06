# python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name project-lunar-lander --seed 1
# CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
#     --exp_name project-lunar-lander-vanilla --seed 3
CUDA_VISIBLE_DEVICES=7 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 \
    --exp_name project-lunar-lander \
    --seed 3 \
    --rho_explore \
    --rho 0.05 \ # perturbation bound
    --lambda 1 \ # number of steps in the mini-rollout
    --rho_sample 5 # number of samples in p-norm ball 

# CUDA_VISIBLE_DEVICES=6 (later run) --lambda 5, --rho_sample 5
# CUDA_VISIBLE_DEVICES=5 (early run) --lambda 1, --rho_sample 5