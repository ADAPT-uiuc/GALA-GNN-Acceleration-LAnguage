import os
import time

import math
import argparse
import subprocess

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Generator')

    parser.add_argument("--pw_abcd", type=str, default="1000",
                        help="Make A power-law in RMAT")
    parser.add_argument("--rat_pw", type=int, default=1,
                        help="Ratio of the number of edges of a power-law node.")
    parser.add_argument("--nodes", type=int, default=10000,
                        help="Number of nodes in the generated graph.")
    parser.add_argument("--mul_edges", type=int, default=10,
                        help="Multiplication of edges compared to the nodes.")

    parser.add_argument("--rmatp", type=str, default="../../utils/third_party/parmat/Release/PaRMAT",
                        help="Location of RMAT")
    parser.add_argument("--outp", type=str, default="/home/damitha/PycharmProjects/generate/",
                        help="Output path")
    args = parser.parse_args()
    print(args)

    parmat_path = args.rmatp
    current_path = args.outp

    inp_prefix = current_path + "coo_"
    inp_suffix = ".coo"

    n = args.nodes
    em = args.mul_edges

    m = [int(x) for x in args.pw_abcd]
    r = (1, args.rat_pw)

    cg = []

    sum_m = m[0] + m[1] + m[2] + m[3]
    less_sum_m = 4 - sum_m

    sum_r = r[0] * less_sum_m + r[1] * sum_m
    inv_sum_r = 1 / sum_r

    main_per = inv_sum_r * r[1]
    rest_per = inv_sum_r * r[0]

    e = n * em  # the total number of edges

    for mi in m:
        if mi == 1:
            cg.append("{rati:.3f}".format(rati=main_per))
        else:
            cg.append("{rati:.3f}".format(rati=rest_per))

    inp_name = str(n) + "_" + str(e) + "_" + cg[0] + "_" + cg[1] + "_" + cg[2]
    inp_path = inp_prefix + inp_name
    coo_inp_path = inp_path + inp_suffix
    res1 = os.system(parmat_path + " -nVertices " + str(n) + " -nEdges "
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
        print("Failed to generate graph:", inp_name)
    else:
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
        with open(inp_prefix + "Adj_src.npy", 'wb') as file_src:
            np.save(file_src, src_node.astype(np.uint32))

        with open(inp_prefix + "Adj_dst.npy", 'wb') as file_dst:
            np.save(file_dst, dst_node.astype(np.uint32))

        os.system("rm " + coo_inp_path)
