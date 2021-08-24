# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import pandas


def merge(output_csv: str, how: str, base_csv: str, *csv_suffix_list):
    if len(csv_suffix_list) % 2 != 0:
        raise RuntimeError("List of CSVs and column suffixes must be of type csv1.csv suffix1 csv2.csv suffix2 ...")
    csv_files = []
    suffixes = []
    for i in range(len(csv_suffix_list)):
        if i % 2 == 0:
            csv_files.append(csv_suffix_list[i])
        else:
            suffixes.append(csv_suffix_list[i])
    df = pandas.read_csv(base_csv)
    for i in range(len(csv_files)):
        df_csv = pandas.read_csv(csv_files[i])
        df_csv.rename(columns=dict([(c, f"{c} {suffixes[i]}") for c in df_csv.columns if c != "Name"]), inplace=True)
        df = df.merge(right=df_csv, how=how, on="Name")
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    fire.Fire(merge)
