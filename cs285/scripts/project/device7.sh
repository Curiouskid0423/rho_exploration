CUDA_VISIBLE_DEVICES=7 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run7 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 20 --seed 1
CUDA_VISIBLE_DEVICES=7 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run7 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 20 --seed 2
CUDA_VISIBLE_DEVICES=7 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run7 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 20 --seed 3