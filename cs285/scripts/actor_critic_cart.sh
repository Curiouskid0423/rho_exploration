# CUDA_VISIBLE_DEVICES=3 
python cs285/scripts/run_hw3_actor_critic.py \
    --env_name CartPole-v0 -n 100 -b 1000 \
    --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
# CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name CartPole-v0 -n 100 -b 1000 \
#     --exp_name q4_100_1 -ntu 100 -ngsptu 1
# CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name CartPole-v0 -n 100 -b 1000 \
#     --exp_name q4_1_100 -ntu 1 -ngsptu 100