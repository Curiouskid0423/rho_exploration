import glob
import tensorflow as tf
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    'blue', 'red', 'grey', 'orange', 'green', 
    'purple', 'olive', 'cyan', 'brown', 'pink', 
    'lime', 'black'
    ]
PLOT_FORMATS = [
        (line_style, color)
        for color in COLORS
    for line_style in ['-', '--', '-.']
]

def get_section_results(file, y_axis='Eval_AverageReturn'):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == y_axis:
                Y.append(v.simple_value)
    return X, Y

def plot_multiple_runs(dict_of_events: dict, title: str, file_name: str):
    X, Y, plot_data = {}, {}, {}
    for key, list_of_events in dict_of_events.items():
        X[key], Y[key] = [], []
        for event in list_of_events:
            eventfile = glob.glob(event)[0]
            x, y = get_section_results(eventfile, y_axis='Train_AverageReturn')
            if len(X[key]) > 0:
                assert len(X[key][-1]) == len(x), \
                    f"X axis length mismatch. Found files with {len(X[key][-1])} and {len(x)} steps."
            X[key].append(x)
            Y[key].append(y)

        Y[key] = np.array(Y[key])
        x_axis = X[key][0][:-1]
        plot_data[key] = {
            'x_axis': x_axis,
            'mean': Y[key].mean(axis=0),
            'min': Y[key].min(axis=0),
            'max': Y[key].max(axis=0)
        }
    
    # Code for plotting
    plt.figure(figsize=(20, 16))
    for idx, (name, data) in enumerate(plot_data.items()):
        line, color = PLOT_FORMATS[idx]
        plt.plot(
            data['x_axis'], data['mean'], marker='o', 
            linestyle=line, color=color, markersize=5, label=name
        )
        # FIXME: instead of `min` or `max`, change this to -0.5 STD and +0.5 STD
        plt.fill_between(x_axis, data['min'], data['max'], color=color, alpha=0.2)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Average Return', fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(file_name)
        
def all_combinations(list_of_vars: List[Tuple]):
    """
    Recursively list out all combinations of variables
    """
    def traverse(list_of_vars, index, curr):
        if index == len(list_of_vars):
            return curr
        result = []
        assert isinstance(list_of_vars[index][1], list), \
            "variables have to be in (name, list of var) format."
        for var in list_of_vars[index][1]:
            choice = traverse(list_of_vars, index+1, curr+[var])
            if len(choice) > 0 and isinstance(choice[0], list):
                result.extend(choice)
            else:
                result.append(choice)
        return result

    return traverse(list_of_vars, index=0, curr=[])



if __name__ == '__main__':
    """
    Variables:
        rho = [.03, .05, .07, .1]
        lambda = [1, 10]
        sample_heuristics = ['max', 'mode']     # sample_threshold=50%
        rho_sample = [10, 20, 30]
    Comparisons:
        1. lambda==1. Study the effect of `rho`, `heuristics`, and `rho_sample`
           (split into 2 graphs -- max vs mode)
        2. lambda==10. Study the effect of `rho` and `rho_sample`
        3. Study the effect of `heuristics` (with lambda==1)
    """
    # Algorithmically generate (key, value) pairs
    variables = [
        ('rho', ['3e-2', '5e-2', '7e-2']),
        ('rho_sample', [10, 20, 30]),
        ('sample_heuristics', ['max']),
        ('lambda', [10])
    ]
    combinations = all_combinations(variables)
    file_names = {
        f'rho{e[0]}_sample{e[1]}_{e[2]}': [
            f'data/lunar-lander-v3_ddqn_rho{e[0]}_sample{e[1]}_lambda{e[3]}_lr1e-3_{e[2]}_seed{seed}_fifty-eps-split/events*'
            for seed in range(1, 4)
        ]
        for e in combinations
    }

    dict_of_events = {
        **file_names,
        'baseline': [
            'data/lunar-lander-v3_vanilla_linear-explore_seed1/events*',
            'data/lunar-lander-v3_vanilla_linear-explore_seed2/events*',
            'data/lunar-lander-v3_vanilla_linear-explore_seed3/events*'
        ],
        'ddqn_baseline': [
            'data/lunar-lander-v3_vanilla_ddqn_linear-explore_seed1/events*',
            'data/lunar-lander-v3_vanilla_ddqn_linear-explore_seed2/events*',
            'data/lunar-lander-v3_vanilla_ddqn_linear-explore_seed3/events*',
            'data/lunar-lander-v3_vanilla_ddqn_linear-explore_seed4/events*',
            'data/lunar-lander-v3_vanilla_ddqn_linear-explore_seed5/events*',
        ]
    }

    plot_multiple_runs(
        dict_of_events=dict_of_events,
        title='DDQN on Lunar Lander. Study the effect of `rho`, `heuristics` and `rho_sample` (lambda=10)',
        file_name='plot_lander_ddqn_linear-exp_lr1e-3_lambda10_max.png')