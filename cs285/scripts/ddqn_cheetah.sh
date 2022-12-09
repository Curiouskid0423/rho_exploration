# add mujoco (e.g. half cheetah) to DQNAgent's list of environments
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name HalfCheetah-v4 \
    --exp_name cheetah-vanilla_ddqn_linear-explore_seed2 \
    --double_q --seed 1
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw3_dqn.py \
    --env_name HalfCheetah-v4 \
    --exp_name cheetah-vanilla_ddqn_linear-explore_seed2 \
    --double_q --seed 2