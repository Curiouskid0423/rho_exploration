# add mujoco (e.g. half cheetah) to DQNAgent's list of environments
# this file tests perturbation
python cs285/scripts/run_hw3_dqn.py \
    --env_name HalfCheetah-v4 --exp_name project-cheetah_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 1
