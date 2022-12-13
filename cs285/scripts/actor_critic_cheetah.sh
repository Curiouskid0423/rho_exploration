# python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name HalfCheetah-v4 \
#     --ep_len 150 \
#     --discount 0.99 \
#     --scalar_log_freq 1 \
#     -n 150 -l 2 -s 32 -b 30000 -eb 150 \
#     -lr 0.01 \
#     --exp_name project_ac_cheetah_baseline -ntu 10 -ngsptu 10 \
#     --seed 1
# CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name HalfCheetah-v4 \
#     --ep_len 150 \
#     --discount 0.99 \
#     --scalar_log_freq 1 \
#     -n 150 -l 2 -s 32 -b 30000 -eb 150 \
#     -lr 0.01 \
#     --exp_name project_ac_cheetah_baseline -ntu 10 -ngsptu 10 \
#     --seed 2