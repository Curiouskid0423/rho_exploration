# CUDA_VISIBLE_DEVICES=8 python cs285/scripts/run_hw3_dqn.py \
#     --env_name LunarLander-v3 \
#     --exp_name project-lunar-lander_ddqn_debug \
#     --double_q \
#     --seed 1 \
#     --rho_explore --rho .05 --heuristics 'max' --lambda 1 --rho_sample 10
# CUDA_VISIBLE_DEVICES=9 python cs285/scripts/run_hw3_dqn.py \
#     --env_name LunarLander-v3 \
#     --exp_name project-lunar-lander_ddqn_debug \
#     --double_q \
#     --seed 1 \
#     --rho_explore --rho .05 --heuristics 'max' --lambda 3 --rho_sample 10
    