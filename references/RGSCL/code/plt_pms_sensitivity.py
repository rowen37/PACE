import matplotlib.pyplot as plt
import config
import numpy as np
cfg = config.CONFIG()
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['Arial']


def plot_hist(data_list, x_label):
    data_arr = np.array(data_list)
    labels = data_arr[:, 0]
    data = data_arr[:, 1:]
    colors1 = ['#30688D' for _ in range(len(data[0]))]
    colors2 = ['#35B777' for _ in range(len(data[0]))]
    colors3 = ['#F8E620' for _ in range(len(data[0]))]

    bar_width = .2
    bar_positions1 = np.arange(len(labels))
    bar_positions2 = bar_positions1 + bar_width
    bar_positions3 = bar_positions2 + bar_width
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.bar(bar_positions1, data[:, 0], width=bar_width, label='Data 1', color=colors1)
    ax.bar(bar_positions2, data[:, 1], width=bar_width, label='Data 2', color=colors2)
    ax.bar(bar_positions3, data[:, 2], width=bar_width, label='Data 3', color=colors3)
    ax.set_xticks(bar_positions2, [int(_) for _ in labels], fontsize=14)
    ax.set_ylim(.4, .8)
    ax.set_yticks([.4, .5, .6, .7, .8], [.4, .5, .6, .7, .8], fontsize=14)

    fig.savefig(f'{cfg.filepath}/figure/{x_label}.png', dpi=600)
    

if __name__ == '__main__':
    h_ls = [(1, .7253, .5215, .6247),
            (2, .7332, .5379, .6381),
            (4, .7412, .5432, .6474),
            (8, .7297, .5312, .6328),
            ]
    d_ls = [(16, .6824, .4647, .5631),
            (32, .7192, .5025, .6030),
            (64, .7324, .5261, .6293),
            (128, .7412, .5432, .6474),
            (256, .7408, .5364, .6437),
            ]
    lamda = [(.1, .7102, .5194, .6210),
             (.3, .7150, .5259, .6298),
             (.5, .7385, .5404, .6418),
             (.7, .7412, .5432, .6474),
             (.9, .7252, .5306, .6362)
             ]
    
    plot_hist(h_ls, 'pms_sensitivity_head')