python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 --exp_name project-lunar-lander_run0 \
    --rho_explore --rho .05 --lambda 1 --rho_sample 10 --seed 1

""" (54 runs) 12/6/2022
    lr  =        [5e-4, 1e-3]
    rho =        [.05, .1, .5]
    lambda =     [1]
    rho_sample = [10, 20, 30]
    seed =       [1, 2, 3]
    sample_heuristics = 'mode'
"""

""" (16 * 3 * 3 = 144 runs) 12/7/2022
    >> eventually 4*2*3*3 + 3*3*3 = 99 runs due to training time constraint
    
    --double_q
    --linear_explore_schedule(0.5, 0.)
    --lr = 1e-3  
    
    >> Experiment Set 1 -- lambda=1
    rho = [.03, .05, .07, .1]
    sample_heuristics = ['max', 'mode']     # sample_threshold=50%
    rho_sample = [10, 20, 30]
    seed = [1, 2, 3]

    >> Experiment Set 2 -- lambda=10
    rho = [.03, .05, .07]
    sample_heuristics = ['max']             # sample_threshold=50%
    rho_sample = [10, 20, 30]
    seed = [1, 2, 3]
    
"""
# CUDA_VISIBLE_DEVICES=0 --rho .03  --heuristics 'max'  --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .05  --heuristics 'max'  --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .07  --heuristics 'max'  --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .1   --heuristics 'max'  --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .03  --heuristics 'mode' --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .05  --heuristics 'mode' --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .07  --heuristics 'mode' --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .1   --heuristics 'mode' --lambda 1,  --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .03  --heuristics 'max'  --lambda 10, --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .05  --heuristics 'max'  --lambda 10, --rho_sample 10 --seed 1
# CUDA_VISIBLE_DEVICES=0 --rho .07  --heuristics 'max'  --lambda 10, --rho_sample 10 --seed 1
# (skipped) CUDA_VISIBLE_DEVICES=0 --rho .1   --heuristics 'max'  --lambda 10, --rho_sample 10 --seed 1
# (skipped) CUDA_VISIBLE_DEVICES=0 --rho .03  --heuristics 'mode' --lambda 10, --rho_sample 10 --seed 1
# (skipped) CUDA_VISIBLE_DEVICES=0 --rho .05  --heuristics 'mode' --lambda 10, --rho_sample 10 --seed 1
# (skipped) CUDA_VISIBLE_DEVICES=0 --rho .07  --heuristics 'mode' --lambda 10, --rho_sample 10 --seed 1
# (skipped) CUDA_VISIBLE_DEVICES=0 --rho .1   --heuristics 'mode' --lambda 10, --rho_sample 10 --seed 1

# CUDA_VISIBLE_DEVICES=1 <repeat> --rho_sample 20 --seed 1
# CUDA_VISIBLE_DEVICES=2 <repeat> --rho_sample 30 --seed 1
# CUDA_VISIBLE_DEVICES=3 <repeat> --rho_sample 10 --seed 2
# CUDA_VISIBLE_DEVICES=4 <repeat> --rho_sample 20 --seed 2
# CUDA_VISIBLE_DEVICES=5 <repeat> --rho_sample 30 --seed 2
# CUDA_VISIBLE_DEVICES=6 <repeat> --rho_sample 10 --seed 3
# CUDA_VISIBLE_DEVICES=7 <repeat> --rho_sample 20 --seed 3
# CUDA_VISIBLE_DEVICES=8 <repeat> --rho_sample 30 --seed 3