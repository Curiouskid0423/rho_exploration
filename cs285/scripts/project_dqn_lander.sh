CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 1

""" (54 runs)
    lr  =        [5e-4, 1e-3]
    rho =        [.05, .1, .5]
    lambda =     [1]
    rho_sample = [10, 20, 30]
    seed =       [1, 2, 3]
"""
# CUDA_VISIBLE_DEVICES=0 --rho .05 --lambda 1, --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .05 --lambda 1, --rho_sample 10 --seed 2
# CUDA_VISIBLE_DEVICES=0 --rho .05 --lambda 1, --rho_sample 10 --seed 3
# CUDA_VISIBLE_DEVICES=1 --rho .05 --lambda 1, --rho_sample 20 --seed 1
# CUDA_VISIBLE_DEVICES=1 --rho .05 --lambda 1, --rho_sample 20 --seed 2
# CUDA_VISIBLE_DEVICES=1 --rho .05 --lambda 1, --rho_sample 20 --seed 3
# CUDA_VISIBLE_DEVICES=2 --rho .05 --lambda 1, --rho_sample 30 --seed 1
# CUDA_VISIBLE_DEVICES=2 --rho .05 --lambda 1, --rho_sample 30 --seed 2
# CUDA_VISIBLE_DEVICES=2 --rho .05 --lambda 1, --rho_sample 30 --seed 3
# CUDA_VISIBLE_DEVICES=3 --rho .1 --lambda 1, --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=3 --rho .1 --lambda 1, --rho_sample 10 --seed 2
# CUDA_VISIBLE_DEVICES=3 --rho .1 --lambda 1, --rho_sample 10 --seed 3
# CUDA_VISIBLE_DEVICES=4 --rho .1 --lambda 1, --rho_sample 20 --seed 1
# CUDA_VISIBLE_DEVICES=4 --rho .1 --lambda 1, --rho_sample 20 --seed 2
# CUDA_VISIBLE_DEVICES=4 --rho .1 --lambda 1, --rho_sample 20 --seed 3
# CUDA_VISIBLE_DEVICES=5 --rho .1 --lambda 1, --rho_sample 30 --seed 1
# CUDA_VISIBLE_DEVICES=5 --rho .1 --lambda 1, --rho_sample 30 --seed 2
# CUDA_VISIBLE_DEVICES=5 --rho .1 --lambda 1, --rho_sample 30 --seed 3
# CUDA_VISIBLE_DEVICES=6 --rho .5 --lambda 1, --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=6 --rho .5 --lambda 1, --rho_sample 10 --seed 2
# CUDA_VISIBLE_DEVICES=6 --rho .5 --lambda 1, --rho_sample 10 --seed 3
# CUDA_VISIBLE_DEVICES=7 --rho .5 --lambda 1, --rho_sample 20 --seed 1
# CUDA_VISIBLE_DEVICES=7 --rho .5 --lambda 1, --rho_sample 20 --seed 2
# CUDA_VISIBLE_DEVICES=7 --rho .5 --lambda 1, --rho_sample 20 --seed 3
# CUDA_VISIBLE_DEVICES=8 --rho .5 --lambda 1, --rho_sample 30 --seed 1
# CUDA_VISIBLE_DEVICES=8 --rho .5 --lambda 1, --rho_sample 30 --seed 2
# CUDA_VISIBLE_DEVICES=8 --rho .5 --lambda 1, --rho_sample 30 --seed 3