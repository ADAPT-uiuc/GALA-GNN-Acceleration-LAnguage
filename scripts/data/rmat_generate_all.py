import os
import time

import math
import argparse
import subprocess

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Generator')
    parser.add_argument("--data", type=str, default="full",
                        help="Graph set to generate. The full graph dataset or the small one (for SPADE collection).")
    parser.add_argument("--till_npy", action='store_true',
                        help="Create the NPY outputs as well.")
    parser.set_defaults(till_npy=True)

    parser.add_argument("--rmatp", type=str, default="../../utils/third_party/parmat/Release/PaRMAT",
                        help="Location of RMAT")
    parser.add_argument("--outp", type=str, default="../../data_schedule",
                        help="Output path")

    args = parser.parse_args()
    print(args)

    parmat_path = args.rmatp
    current_path = args.outp

    inp_prefix = current_path
    inp_suffix = ".coo"

    # if args.data == 'small':
    #     graphs_at_config = [
    #         ["0.25", "0.25", "0.25"],  # 1:1
    #         ["0.40", "0.20", "0.20"],  # 1:2
    #         ["0.769", "0.077", "0.077"],  # 1:10
    #         ["0.97", "0.01", "0.01"]  # 1:100
    #     ]
    # elif args.data == 'test':
    #     graphs_at_config = [
    #         ["0.25", "0.25", "0.25"],
    #         ["0.769", "0.077", "0.077"],  # 1:10
    #         ["0.4", "0.4", "0.1"],
    #         ["0.1", "0.2", "0.3"],
    #         ["0.4", "0.1", "0.1"],
    #         ["0.1", "0.7", "0.1"],
    #         ["0.97", "0.01", "0.01"]
    #     ]
    # else:
    #     graphs_at_config = [
    #         ["0.25", "0.25", "0.25"],  # 1:1
    #         ["0.40", "0.20", "0.20"],  # 1:2
    #         ["0.625", "0.125", "0.125"],  # 1:5
    #         ["0.769", "0.077", "0.077"],  # 1:10
    #         ["0.943", "0.0189", "0.0189"],  # 1:50
    #         ["0.97", "0.01", "0.01"],  # 1:100
    #         ["0.994", "0.002", "0.002"]  # 1:500
    #     ]
    if args.data == 'small':
        graphs_main = [
            (1, 0, 0, 0),  # Clustered around diagonal
            (1, 1, 0, 0),  # In between
            (1, 1, 1, 0),  # Spread around
        ]
        graphs_ratio = [
            (1, 1),  # 1:1
            (1, 2),  # 1:2
            (1, 10),  # 1:10
            (1, 100)  # 1:100
        ]
        nodes = [10000]
        mul_edges = [2]
    elif args.data == 'test':
        graphs_main = [
            (1, 0, 0, 0),  # Clustered around diagonal
            (1, 1, 0, 0),  # In between
            (1, 1, 1, 0),  # Spread around
        ]
        graphs_ratio = [
            (1, 1),  # 1:1
            (1, 2),  # 1:2
            (1, 10),  # 1:10
            (1, 100)  # 1:100
        ]
        nodes = [10000]
        mul_edges = [100]
    else:
        graphs_main = [
            (1, 0, 0, 0),  # Clustered around diagonal -  Highly concentrated
            (1, 0, 0, 1),  # In between concentrated and spread around
            (1, 1, 0, 1),  # Spread around
            (0, 1, 0, 0),  # Clustered around inverse diagonal -  Highly concentrated
            (0, 1, 1, 0),  # In between concentrated and spread around
            (1, 1, 0, 0),  # Power law - In between concentrated and spread around
            (1, 1, 1, 0),  # Spread around
        ]
        graphs_ratio = [
            (1, 1),  # 1:1
            (1, 2),  # 1:2
            (1, 5),  # 1:5
            (1, 10),  # 1:10
            (1, 50),  # 1:50
            (1, 100),  # 1:100
            (1, 500)  # 1:100
        ]
        nodes = [10000,
                 25000,
                 50000,
                 100000,
                 250000,
                 500000,
                 1000000,
                 2500000,
                 5000000,
                 10000000]
        mul_edges = [2,
                     5,
                     10,
                     25,
                     50,
                     100]

    graphs_at_config = []

    total_fails = 0
    fails = []

    for r in graphs_ratio:
        for m in graphs_main:
            conf = []

            sum_m = m[0] + m[1] + m[2] + m[3]
            less_sum_m = 4 - sum_m

            sum_r = r[0] * less_sum_m + r[1] * sum_m
            inv_sum_r = 1 / sum_r

            main_per = inv_sum_r * r[1]
            rest_per = inv_sum_r * r[0]

            for mi in m:
                if mi == 1:
                    conf.append("{rati:.3f}".format(rati=math.floor(main_per * 100)/100))
                else:
                    conf.append("{rati:.3f}".format(rati=rest_per))
            # print(conf, conf in graphs_at_config)
            if conf not in graphs_at_config:
                graphs_at_config.append(conf)

    for n in nodes:
        for em in mul_edges:
            e = n * em  # the total number of edges

            for cg in graphs_at_config:
                count_fails = 0
                gen_success = False

                inp_name = str(n) + "_" + str(e) + "_" + cg[0] + "_" + cg[1] + "_" + cg[2]
                inp_path = inp_prefix + inp_name
                coo_inp_path = inp_path + inp_suffix

                while count_fails < 5 and not gen_success:
                    res1 = os.system("timeout 600s " + parmat_path + " -nVertices " + str(n) + " -nEdges "
                                     + str(e) + " -output "
                                     + coo_inp_path
                                     + " -a " + cg[0]
                                     + " -b " + cg[1]
                                     + " -c " + cg[2]
                                     # + " -noEdgeToSelf"
                                     # + " -noDuplicateEdges"
                                     + " -sorted"
                                     # + " -undirected"
                                     )
                    if res1 != 0:
                        count_fails += 1
                    else:
                        gen_success = True

                if gen_success:
                    if args.till_npy:
                        coo_src = np.empty(e)
                        coo_dst = np.empty(e)

                        e_i = 0
                        with open(coo_inp_path, "r") as file:
                            for line in file:
                                src, dst = list(map(int, (line.strip().split())))
                                coo_src[e_i] = src
                                coo_dst[e_i] = dst
                                e_i += 1
                                # print(src, dst)
                                # break

                        src_node = np.append([n, n], coo_src)
                        dst_node = coo_dst
                        with open(inp_path + "_src.npy", 'wb') as file_src:
                            np.save(file_src, src_node.astype(np.uint32))

                        with open(inp_path + "_dst.npy", 'wb') as file_dst:
                            np.save(file_dst, dst_node.astype(np.uint32))

                        os.system("rm " + coo_inp_path)
                else:
                    total_fails += 1
                    fails.append(inp_name)

    print("Total fails:", total_fails)
    print(fails)
