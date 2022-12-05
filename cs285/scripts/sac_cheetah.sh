CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw3_sac.py \
    --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.99 --scalar_log_freq 1500 \
    -n 200000 -l 2 -s 256 -b 1500 -eb 1500 \
    -lr 0.0003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah \
    --seed 1