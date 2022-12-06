# python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name project-lunar-lander --seed 1
# CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
#     --exp_name project-lunar-lander-vanilla --seed 3
CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 \
    --exp_name project-lunar-lander \
    --seed 3 \
    --rho_explore \
    --rho 0.05 \
    --lambda 1 \
    --rho_sample 10 

# CUDA 0: rho_sample 20 (early run)
# CUDA 1: rho_sample 10  (later run)
