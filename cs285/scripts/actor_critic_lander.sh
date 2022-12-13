# CUDA_VISIBLE_DEVICES=8 python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name LunarLander-v3 \
#     --exp_name project_ac_lunar-lander_baseline \
#     --ep_len 1000 \
#     --discount 0.99 \
#     -b 40000 \
#     -n 100 \
#     --num_agent_train_steps_per_iter 2 \
#     --num_critic_updates_per_agent_update 10 \
#     --seed 1
# CUDA_VISIBLE_DEVICES=9 python cs285/scripts/run_hw3_actor_critic.py \
#     --env_name LunarLander-v3 \
#     --exp_name project_ac_lunar-lander_baseline \
#     --ep_len 1000 \
#     --discount 0.99 \
#     -b 40000 \
#     -n 100 \
#     -lr 0.01 \
#     --num_agent_train_steps_per_iter 1 \
#     --num_critic_updates_per_agent_update 10 \
#     --seed 1