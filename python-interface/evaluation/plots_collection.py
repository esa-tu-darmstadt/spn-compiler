# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas

cmu_serif = {'fontname': 'CMU Serif', 'size': 14}
cmu_sans_serif = {'fontname': 'CMU Sans Serif', 'size': 14}

speaker_color = '#009CDA'
ratspn_color = '#ff6325'

low = (0.094, 0.310, 0.635)
high = (0.565, 0.392, 0.173)
medium_grey = '#555555'
medium_red = '#cc0000'

marker = "x"


def is_speaker(name: str):
    return len(name) <= 5


def two_colors(dataframe):
    colors = []
    for index, row in dataframe.iterrows():
        if is_speaker(row['Name']):
            colors.append(speaker_color)
        else:
            colors.append(ratspn_color)
    return colors


def gradient(dataframe, rgb_start, rgb_end, column=None):
    colors = []
    if column is None:
        delta_r = (rgb_end[0] - rgb_start[0]) / len(dataframe)
        delta_g = (rgb_end[1] - rgb_start[1]) / len(dataframe)
        delta_b = (rgb_end[2] - rgb_start[2]) / len(dataframe)
        for i in range(len(dataframe)):
            colors.append(
                (np.round((rgb_start[0] + i * delta_r)),
                 np.round((rgb_start[1] + i * delta_g)),
                 np.round((rgb_start[2] + i * delta_b)))
            )
    else:
        column_min = column.min()
        interval = column.max() - column_min
        delta_r = (rgb_end[0] - rgb_start[0]) / interval
        delta_g = (rgb_end[1] - rgb_start[1]) / interval
        delta_b = (rgb_end[2] - rgb_start[2]) / interval
        for i in range(len(column)):
            colors.append(
                (np.round((rgb_start[0] + (column.iloc[i] - column_min) * delta_r)),
                 np.round((rgb_start[1] + (column.iloc[i] - column_min) * delta_g)),
                 np.round((rgb_start[2] + (column.iloc[i] - column_min) * delta_b)))
            )
    return [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def include_optimal_arith_reduction(x_axis, vector_width=8):
    min = np.min(x_axis)
    max = np.max(x_axis)
    t = np.linspace(start=min, stop=max, num=len(x_axis))
    plt.plot(t, [100 / ((100 - x) + (x / vector_width)) for x in t], color=medium_grey, label="Optimum (ohne shuffles)",
             zorder=2.0)


def include_linear_line(ax, x_axis, color, label, zorder):
    min = np.min(x_axis)
    max = np.max(x_axis)
    return ax.axline(xy1=[min, min], xy2=[max, max], color=color, label=label, zorder=zorder)


def draw_constant_line(ax, value, color, label, zorder):
    interval = ax.get_xlim()
    t = np.linspace(start=interval[0], stop=interval[1], num=100)
    ax.plot(t, [value] * 100, color=color, label=label, zorder=zorder)



def make_plots(csv_file: str, sans_serif: bool, number: int):
    if sans_serif:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [cmu_sans_serif['fontname']]
        plt.rcParams['font.size'] = cmu_sans_serif['size']
    else:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [cmu_serif['fontname']]
        plt.rcParams['font.size'] = cmu_sans_serif['size']

    df = pandas.read_csv(csv_file)
    df = df[df['#profitable iterations dfs'] > 0]
    speaker_df = df[df['Name'].str.len() <= 5]
    ratspn_df = df[df['Name'].str.len() > 5]

    # ops covered -> (arithmetic ops normal / arithmetic ops dfs)
    plot_number = 1
    if number == plot_number:
        x_speaker = speaker_df['% function ops dead after iteration 0 dfs']
        y_speaker = speaker_df['#arithmetic ops normal'] / speaker_df['#arithmetic ops dfs']
        plt.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        x_ratspn = ratspn_df['% function ops dead after iteration 0 dfs']
        y_ratspn = ratspn_df['#arithmetic ops normal'] / ratspn_df['#arithmetic ops dfs']
        plt.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        include_optimal_arith_reduction(df['% function ops dead after iteration 0 dfs'])
        plt.grid(zorder=0.0)
        plt.xlabel(xlabel='Abgedeckte Operationen [%]')
        plt.ylabel(ylabel='#arithmetisch unvektorisiert / #arithmetisch DFS')
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 2, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.show()

    # ops covered -> (arithmetic ops normal / arithmetic ops dfs)
    plot_number = plot_number + 1
    if number == plot_number:
        filtered_speaker = speaker_df[speaker_df['% function ops dead after iteration 0 dfs'] > 92]
        x_speaker = filtered_speaker['% function ops dead after iteration 0 dfs']
        y_speaker = filtered_speaker['#arithmetic ops normal'] / filtered_speaker['#arithmetic ops dfs']
        plt.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        filtered_ratspn = ratspn_df[ratspn_df['% function ops dead after iteration 0 dfs'] > 92]
        x_ratspn = filtered_ratspn['% function ops dead after iteration 0 dfs']
        y_ratspn = filtered_ratspn['#arithmetic ops normal'] / filtered_ratspn['#arithmetic ops dfs']
        plt.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        include_optimal_arith_reduction(filtered_ratspn['% function ops dead after iteration 0 dfs'])
        plt.grid(zorder=0.0)
        plt.xlabel(xlabel='Abgedeckte Operationen [%]')
        plt.ylabel(ylabel='#arithmetisch unvektorisiert / #arithmetisch DFS')
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 2, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.show()

    # ops covered -> SLP graph overhead
    plot_number = plot_number + 1
    if number == plot_number:
        x_speaker = speaker_df['#unique arithmetic op in graph 0 dfs'] / speaker_df['#superwords in graph 0 dfs']
        y_speaker = speaker_df['% function ops dead after iteration 0 dfs']
        plt.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        x_ratspn = ratspn_df['#unique arithmetic op in graph 0 dfs'] / ratspn_df['#superwords in graph 0 dfs']
        y_ratspn = ratspn_df['% function ops dead after iteration 0 dfs']
        plt.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        plt.grid(zorder=0.0)
        plt.xlabel(xlabel='Verhältnis Operationen : Vektoren im SLP-Graph')
        plt.ylabel(ylabel='Abgedeckte Operationen [%]')
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.show()

    # original kernel size -> kernel size DFS by coverage
    plot_number = plot_number + 1
    if number == plot_number:
        x = df['#ops normal']
        y = df['#ops dfs']
        color_values = df['% function ops dead after iteration 0 dfs']
        fig, ax = plt.subplots()
        plot = ax.scatter(x, y, c=color_values, cmap=make_colormap([low, high]), marker=marker, label="Kernels DFS", zorder=3.0, vmin=0, vmax=100)
        colorbar = fig.colorbar(plot)
        colorbar.set_label('Abgedeckte Operationen [%]')
        line = include_linear_line(ax, df['#ops normal'], medium_red, "Unverändert", 2.0)
        ax.grid(zorder=0.0)
        ax.set_xlabel(xlabel='Instruktionen unvektorisiert')
        ax.set_ylabel(ylabel='Instruktionen DFS')
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend([plot, line], [plot.get_label(), line.get_label()])
        plt.show()

    # original kernel size -> kernel size BFS by coverage
    plot_number = plot_number + 1
    if number == plot_number:
        x = df['#ops normal']
        y = df['#ops bfs']
        color_values = df['% function ops dead after iteration 0 bfs']
        fig, ax = plt.subplots()
        plot = ax.scatter(x, y, c=color_values, cmap=make_colormap([low, high]), marker=marker, label="Kernels BFS", zorder=3.0, vmin=0, vmax=100)
        colorbar = fig.colorbar(plot)
        colorbar.set_label('Abgedeckte Operationen [%]')
        line = include_linear_line(ax, df['#ops normal'], medium_red, "Unverändert", 2.0)
        ax.grid(zorder=0.0)
        ax.set_xlabel(xlabel='Instruktionen unvektorisiert')
        ax.set_ylabel(ylabel='Instruktionen BFS')
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend([plot, line], [plot.get_label(), line.get_label()])
        plt.show()

    # BFS kernel size -> DFS kernel size
    plot_number = plot_number + 1
    if number == plot_number:
        x_speaker = speaker_df['#ops bfs']
        y_speaker = speaker_df['#ops bfs'] / speaker_df['#ops dfs']
        scatter1 = plt.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        x_ratspn = ratspn_df['#ops bfs']
        y_ratspn = ratspn_df['#ops bfs'] / ratspn_df['#ops dfs']
        scatter2 = plt.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        plt.grid(zorder=0.0)
        plt.xlabel(xlabel='Instruktionen BFS')
        plt.ylabel(ylabel='Instruktionen BFS / Instruktionen DFS')
        plt.xscale("log")
        plt.legend([scatter1, scatter2], [scatter1.get_label(), scatter2.get_label()])
        plt.show()

    # TODO
    plot_number = plot_number + 1
    if number == plot_number:
        x = (df['#ops normal']) / (df['#ops dfs'])
        y = df['execution time total (ns) normal'] / df['execution time total (ns) dfs']
        color_values = df['% function ops dead after iteration 0 dfs']
        fig, ax = plt.subplots()
        plot = ax.scatter(x, y, c=color_values, cmap=make_colormap([low, high]), marker=marker, zorder=3.0, vmin=0, vmax=100)
        colorbar = fig.colorbar(plot)
        colorbar.set_label('Abgedeckte Operationen [%]')
        ax.grid(b=True, which='both', zorder=0.0)
        loc = plticker.MultipleLocator(base=1.0)
        ax.yaxis.set_major_locator(loc)
        # ax.set_xlabel(xlabel='Abgedeckte Operationen [%]')
        ax.set_ylabel(ylabel='Speedup')
        #ax.set_xscale("log")
        # plt.yscale("log")
        plt.show()

    # TODO: arithmetic call targets checken
    plot_number = plot_number + 1
    if number == plot_number:
        x = (df['#lo_spn.add pre dfs']) * (df['% function ops dead after iteration 0 dfs'] / 100) / (df['#lospn ops'])
        y = df['execution time total (ns) normal'] / df['execution time total (ns) dfs']
        color_values = df['% function ops dead after iteration 0 dfs']
        fig, ax = plt.subplots()
        plot = ax.scatter(x, y, c=color_values, cmap=make_colormap([low, high]), marker=marker, zorder=3.0, vmin=0, vmax=100)
        colorbar = fig.colorbar(plot)
        colorbar.set_label('Abgedeckte Operationen [%]')
        ax.grid(b=True, which='both', zorder=0.0)
        loc = plticker.MultipleLocator(base=1.0)
        ax.yaxis.set_major_locator(loc)
        # ax.set_xlabel(xlabel='Abgedeckte Operationen [%]')
        ax.set_ylabel(ylabel='Speedup')
        #ax.set_xscale("log")
        # plt.yscale("log")
        plt.show()

    # original kernel size -> kernel size DFS speaker/RAT-SPN distinction
    plot_number = plot_number + 1
    if number == plot_number:
        fig, ax = plt.subplots()
        x_speaker = speaker_df['#ops normal']
        y_speaker = speaker_df['#ops dfs']
        scatter1 = ax.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        x_ratspn = ratspn_df['#ops normal']
        y_ratspn = ratspn_df['#ops dfs']
        scatter2 = ax.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        line = include_linear_line(ax, df['#ops normal'], medium_grey, "Unverändert", 2.0)
        ax.grid(zorder=0.0)
        ax.set_xlabel(xlabel='Instruktionen unvektorisiert')
        ax.set_ylabel(ylabel='Instruktionen DFS')
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend([scatter1, scatter2, line], [scatter1.get_label(), scatter2.get_label(), line.get_label()])
        plt.show()


if __name__ == '__main__':
    fire.Fire(make_plots)
