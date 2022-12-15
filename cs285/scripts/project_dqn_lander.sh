""" (15 * 12 + 36 = 216 runs) 12/09/2022

    <DQN>
    (1)
        --double_q
        --linear_explore_schedule(0.5, 0.,)
        * rho =                   [.03, .05, .07]
        * rho_sample =            [10, 30]
        * rho_exp_threshold =     [.2, .5]
        lr  =                   [1e-3]
        sample_heuristics =     ['max']     # if time allows, do `mode`
        rho_exp_interval =      [10, 50, 100]
        seed =                  [1, 2, 3]
        lambda =                [1, 5]
        >>  just 2 random seeds for the `lambda=5` runs, making it 6 runs per machine, making
        >>  3 * 2 * 2 * 3 * 3 + 3 * 2 * 2 * 3 * 2 = 108 + 72 = 180 runs
    (2)
        further studied `rho_exp_interval=1` case with the following hyperparameters
        >>  lambda=[1, 5]  rho=[.03, .05, .07]  rho_sample=10  rho_exp_threshold=[.2, .5]
        >>  2*3*1*2 * 3 seeds = 36 runs
    
    Naming:

    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval10_lmbd1_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval10_lmbd1_seed2
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval10_lmbd1_seed3
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval50_lmbd1_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval50_lmbd1_seed2
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval50_lmbd1_seed3
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval100_lmbd1_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval100_lmbd1_seed2
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval100_lmbd1_seed3

    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval10_lmbd5_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval10_lmbd5_seed2
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval50_lmbd5_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval50_lmbd5_seed2
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval100_lmbd5_seed1
    lander-v3_ddqn_schedule_rho<X>_sample<Y>_threshold<Z>_interval100_lmbd5_seed2


"""
# 
# 180 / 12 -> each machine has 15 runs (9 for lambda=1; 6 for lambda=5 by just running 2 seeds)
#
# < windigo > 
## CUDA_VISIBLE_DEVICES=0   <some_hparams>   --rho .03 --rho_sample 30 --rho_exp_threshold .2
## CUDA_VISIBLE_DEVICES=1   <some_hparams>   --rho .03 --rho_sample 30 --rho_exp_threshold .5
# CUDA_VISIBLE_DEVICES=2   <some_hparams>   --rho .05 --rho_sample 10 --rho_exp_threshold .2
# CUDA_VISIBLE_DEVICES=3   <some_hparams>   --rho .05 --rho_sample 10 --rho_exp_threshold .5
## CUDA_VISIBLE_DEVICES=4   <some_hparams>   --rho .05 --rho_sample 30 --rho_exp_threshold .2
## CUDA_VISIBLE_DEVICES=5   <some_hparams>   --rho .05 --rho_sample 30 --rho_exp_threshold .5
# CUDA_VISIBLE_DEVICES=6   <some_hparams>   --rho .07 --rho_sample 10 --rho_exp_threshold .2
# CUDA_VISIBLE_DEVICES=7   <some_hparams>   --rho .07 --rho_sample 10 --rho_exp_threshold .5
## CUDA_VISIBLE_DEVICES=8   <some_hparams>   --rho .07 --rho_sample 30 --rho_exp_threshold .2
## CUDA_VISIBLE_DEVICES=9   <some_hparams>   --rho .07 --rho_sample 30 --rho_exp_threshold .5
# 
# < cthulhu2>
# CUDA_VISIBLE_DEVICES=0   <some_hparams>   --rho .03 --rho_sample 10 --rho_exp_threshold .2
# CUDA_VISIBLE_DEVICES=1   <some_hparams>   --rho .03 --rho_sample 10 --rho_exp_threshold .5
#
# < leviathan > 
# (1) 5 vanilla baseline of DDQN on Lunar Lander
# (2) try rho_exp_interval = 1 (effectively 2 steps)
#     repeat the following for 3 seeds, each machine runs a seed (12 runs per machine)
#     --lambda=[1, 5] --rho=[.03, .05, .07]  --rho_sample=10 --rho_exp_threshold=[.2, .5]


""" (54 runs) 12/06/2022
    lr  =        [5e-4, 1e-3]
    rho =        [.05, .1, .5]
    lambda =     [1]
    rho_sample = [10, 20, 30]
    seed =       [1, 2, 3]
    sample_heuristics = 'mode'
"""

""" (144 runs) 12/07/2022
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