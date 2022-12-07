CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 1
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 2
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 3