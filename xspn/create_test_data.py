import argparse
import sys
import random
import bisect

import numpy as np
from spn_parser import load_spn

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type, Sum, Product
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe


# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
def highlight(text: str) -> str:
    return f'\x1b[6;30;42m{text}\x1b[0m'


def custom_mpe(node, data):
    node_values = dict()
    
    def bottom_up(node):
        if isinstance(node, Sum):
            p = 0
            for child, weight in zip(node.children, node.weights):
                bottom_up(child)
                p += node_values[child] * weight
            node_values[node] = p
        elif isinstance(node, Product):
            p = 1
            for child in node.children:
                bottom_up(child)
                p *= node_values[child]
            node_values[node] = p
        elif isinstance(node, Histogram):
            d = data[0][node.scope[0]]
            if np.isnan(d):
                p = 1
            else:
                p = node.densities[bisect.bisect(node.breaks, d) - 1]
            node_values[node] = p

    bottom_up(node)
    #pprint.pprint(node_values)

    result = data.copy()

    def top_down(node):
        if isinstance(node, Sum):
            max_child_value = 0
            max_child = None

            for child, weight in zip(node.children, node.weights):
                if node_values[child] * weight > max_child_value:
                    max_child_value = node_values[child] * weight
                    max_child = child

            top_down(max_child)
        elif isinstance(node, Product):
            for child in node.children:
                top_down(child)
        elif isinstance(node, Histogram):
            if np.isnan(data[0][node.scope[0]]):
                max_idx = np.argmax(np.array(node.densities))
                max_class = node.breaks[max_idx]
                result[0][node.scope[0]] = max_class

    top_down(spn)

    #print(f'custom node values:')
    #for n, val in node_values.items():
    #    print(f'{n.name}: {val}')

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--marginal', action='store_true', help='Compute marginlization on randomly generated inputs.', default=False)
    parser.add_argument('--mpe', action='store_true', help='Compute MPE on randomly generated inputs.', default=False)
    parser.add_argument('--sum_out_prct', type=float, help='The percentage for a given variable to be summed out.', default=0.1)

    parser.add_argument('--samples', type=int, help='The number of sampled to generate.', default=100)
    parser.add_argument('--verbose', help='Additionally print the generated data to the console.', action='store_true', default=True)
    parser.add_argument('--query', type=str, help='The path of the resulting query csv file.')
    parser.add_argument('--output', type=str, help='The path of the resulting output csv file.')
    parser.add_argument('--spn', type=str, help='The path of the .spn file to be read.')
    parser.add_argument('--zero', type=float, help='The percentage that an entry is set to 0.', default=0.5)
    parser.add_argument('--test', action='store_true', help='Use SPFlow to compute the input given by --query and compare it to the results given by --output.', default=False)

    parsed = parser.parse_args()

    if ((not parsed.marginal) and (not parsed.mpe)) or (parsed.marginal and parsed.mpe):
        print('Exactly one of --marginal or --mpe has to be specified!')
        exit()

    spn_path = parsed.spn
    spn, variables_to_index, index_to_min, index_to_max = load_spn(spn_path)

    min_values = [b for _, b in sorted(index_to_min.items())]
    max_values = [b for _, b in sorted(index_to_max.items())]

    sum_out_prct = parsed.sum_out_prct
    zero_prct = parsed.zero
    sample_count = parsed.samples
    verbose = parsed.verbose

    def roll(min_val, max_val):
        zero = random.random() < zero_prct

        if zero:
            return 0

        sum_out = random.random() < sum_out_prct

        if sum_out:
            return max_val
        else:
            return random.randint(min_val, max_val - 1)

    if parsed.mpe:
        def query(data):
            #return mpe(node=spn, input_data=data)
            return custom_mpe(spn, data)
    else:
        def query(data):
            return log_likelihood(node=spn, data=data)


    if parsed.test:
        raise NotImplementedError()

        query_matrix = np.loadtxt(parsed.query, delimiter=';')
        prob_matrix = np.loadtxt(parsed.prob, delimiter=';')

        result = query(query_matrix).reshape(-1)
        
        for r, p in zip(result, prob_matrix):
            print(f'got {r} wanted {p} delta {r - p}')
    else:
        with open(parsed.query, 'w') as query_file, open(parsed.output, 'w') as out_file:
            for _ in range(sample_count):
                random_values = [roll(mn, mx) for mn, mx in zip(min_values, max_values)]
                is_summed_out = [rand == mx for rand, mx in zip(random_values, max_values)]

                # for MPE we have to force at least one entry to be summed out
                if parsed.mpe and not any(is_summed_out):
                    idx = random.randint(0, len(random_values) - 1)
                    random_values[idx] = max_values[idx]
                    is_summed_out[idx] = True

                # SPFlow uses NaN to mark variables that are summed out
                random_values_nan = [np.nan if sum_out else val for sum_out, val in zip(is_summed_out, random_values)]
                #result = np.exp(query(np.array([random_values_nan])))
                # result can be a log-probability or a vector of histogram class values
                if parsed.mpe:
                    result = query(np.array([random_values_nan]))
                else:
                    result = query(np.array([random_values_nan])).item()

                query_line = ';'.join(str(val) for val in random_values)
                query_file.write(query_line + '\n')

                if parsed.mpe:
                    result = result.astype(int)
                    prob_line = ';'.join(str(val) for val in result[0])
                    out_file.write(prob_line + '\n')

                    if verbose:
                        txt_in  = ', '.join(str(val) if not summed_out else highlight(str(val)) for val, summed_out in zip(random_values, is_summed_out))
                        txt_out = ', '.join(str(val) if not summed_out else highlight(str(val)) for val, summed_out in zip(result[0], is_summed_out))
                        print(f'MPE[{txt_in}] = [{txt_out}]')
                else:
                    prob_line = str(result)
                    out_file.write(prob_line + '\n')

                    if verbose:
                        # additionally highlight the marginalized values
                        txt = ', '.join(str(val) if not summed_out else highlight(str(val)) for val, summed_out in zip(random_values, is_summed_out))
                        print(f'P[ {txt} ] ~= {result}')
