CUDA_VISIBLE_DEVICES=9 python cs285/scripts/run_hw3_sac.py \
    --env_name LunarLander-v3 \
    --exp_name debug_test_sac_LunarLander \
    --ep_len 150 \
    --discount 0.99 \
    --scalar_log_freq 1500 \
    -n 400000 -l 2 -s 256 -b 1500 -eb 1500 \
    -lr 0.0003 \
    --init_temperature 0.1 \
    --seed 1