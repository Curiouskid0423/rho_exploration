import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['blue', 'red', 'grey', 'orange', 'green', 'purple', 'pink', 'olive', 'cyan', 'brown']

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

def plot_multiple_runs(dict_of_events: dict):
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
    plt.figure(figsize=(18, 12))
    for idx, (name, data) in enumerate(plot_data.items()):
        plt.plot(
            data['x_axis'], data['mean'], marker='o', 
            linestyle='-', color=COLORS[idx], markersize=5, label=name
        )
        plt.fill_between(x_axis, data['min'], data['max'], color=COLORS[idx], alpha=0.2)
    plt.title('Lunar Lander on DQN, 50-epsilon Split, Linear Schedule (lr=1e-3)')
    plt.legend()
    plt.savefig('plot_with_error_bar.png')
        

if __name__ == '__main__':
    
    dict_of_events = {
        'baseline': [
            'data/lunar-lander-v3_vanilla_linear-explore_seed1/events*',
            'data/lunar-lander-v3_vanilla_linear-explore_seed2/events*',
            'data/lunar-lander-v3_vanilla_linear-explore_seed3/events*'
        ],
        'rho1e-1_sample10_lambda1': [
            'data/lunar-lander-v3_rho1e-1_sample10_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample10_lambda1_lr1e-3_seed2_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample10_lambda1_lr1e-3_seed3_fifty-eps-split/events*',
        ],
        'rho1e-1_sample20_lambda1': [
            'data/lunar-lander-v3_rho1e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ],
        'rho1e-1_sample30_lambda1': [
            'data/lunar-lander-v3_rho1e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho1e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ],
        'rho5e-1_sample10_lambda1': [
            'data/lunar-lander-v3_rho5e-1_sample10_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample10_lambda1_lr1e-3_seed2_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample10_lambda1_lr1e-3_seed3_fifty-eps-split/events*',
        ],
        'rho5e-1_sample20_lambda1': [
            'data/lunar-lander-v3_rho5e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ],
        'rho5e-1_sample30_lambda1': [
            'data/lunar-lander-v3_rho5e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-1_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ],
        'rho5e-2_sample10_lambda1': [
            'data/lunar-lander-v3_rho5e-2_sample10_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample10_lambda1_lr1e-3_seed2_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample10_lambda1_lr1e-3_seed3_fifty-eps-split/events*',
        ],
        'rho5e-2_sample20_lambda1': [
            'data/lunar-lander-v3_rho5e-1_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample20_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ],
        'rho5e-2_sample30_lambda1': [
            'data/lunar-lander-v3_rho5e-2_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
            'data/lunar-lander-v3_rho5e-2_sample30_lambda1_lr1e-3_seed1_fifty-eps-split/events*',
        ]
    }
    plot_multiple_runs(dict_of_events=dict_of_events)