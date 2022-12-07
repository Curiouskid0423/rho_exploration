CUDA_VISIBLE_DEVICES=6 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run6 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 10 --seed 1
CUDA_VISIBLE_DEVICES=6 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run6 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 10 --seed 2
CUDA_VISIBLE_DEVICES=6 python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run6 \
    --rho_explore --rho .5 --lambda 1 --rho_sample 10 --seed 3