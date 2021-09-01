# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas

cmu_serif = {'fontname': 'CMU Serif', 'size': 12}
cmu_sans_serif = {'fontname': 'CMU Sans Serif', 'size': 12}

speaker_color = '#009CDA'
ratspn_color = '#ff6325'

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


def include_optimal_arith_reduction(x_axis, vector_width=8):
    min = np.min(x_axis)
    max = np.max(x_axis)
    t = np.linspace(start=min, stop=max, num=len(x_axis))
    plt.plot(t, [100 / ((100 - x) + (x / vector_width)) for x in t], color='#555555', label="Optimum (ohne shuffles)",
             zorder=2.0)


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
    speaker_df = df[df['Name'].str.len() <= 5]
    ratspn_df = df[df['Name'].str.len() > 5]

    plot_number = 1

    # ops covered -> (arithmetic ops normal / arithmetic ops dfs)
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
    plot_number = plot_number + 1

    # ops covered -> (arithmetic ops normal / arithmetic ops dfs)
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
    plot_number = plot_number + 1

    # ops covered -> SLP graph overhead
    if number == plot_number:
        x_speaker = speaker_df['#unique arithmetic op in graph 0 dfs'] / speaker_df['#superwords in graph 0 dfs']
        y_speaker = speaker_df['% function ops dead after iteration 0 dfs']
        plt.scatter(x_speaker, y_speaker, color=speaker_color, marker=marker, label="speaker", zorder=3.0)
        x_ratspn = ratspn_df['#unique arithmetic op in graph 0 dfs'] / ratspn_df['#superwords in graph 0 dfs']
        y_ratspn = ratspn_df['% function ops dead after iteration 0 dfs']
        plt.scatter(x_ratspn, y_ratspn, color=ratspn_color, marker=marker, label="RAT-SPN", zorder=3.0)
        plt.grid(zorder=0.0)
        plt.xlabel(xlabel='Abgedeckte Operationen [%]')
        plt.ylabel(ylabel='#arithmetisch unvektorisiert / #arithmetisch DFS')
        plt.xlabel(xlabel='Verhältnis Operationen : Vektoren im SLP-Graph')
        plt.ylabel(ylabel='Abgedeckte Operationen [%]')
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.show()
    plot_number = plot_number + 1

    # ops covered (99%+) -> (arithmetic ops normal / arithmetic ops dfs)
    if number == plot_number:
        filtered_df = df[df['% function ops dead after iteration 0 dfs'] > 95]
        x = filtered_df['% function ops dead after iteration 0 dfs']
        y = filtered_df['#arithmetic ops normal'] / filtered_df['#arithmetic ops dfs']
        plt.grid()
        plt.scatter(x, y, color=two_colors(filtered_df), marker=marker)
        plt.xlabel(xlabel='Abgedeckte Operationen [%]')
        plt.ylabel(ylabel='#arithmetisch unvektorisiert / #arithmetisch DFS')
        plt.show()
    plot_number = plot_number + 1


if __name__ == '__main__':
    fire.Fire(make_plots)
