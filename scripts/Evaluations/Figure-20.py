import argparse
import subprocess
import os

build_path = r"../../build/"

graphs = ["Reddit", "Products"]
modes = ["schedule", "input-aware"]

sample_precen = ["1", "2", "5", "10", "20"]

def run(args, logfile, errfile):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile)
    proc.wait()
    logfile.flush()
    errfile.flush()

def run_at(args, logfile, errfile, path):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile, cwd=path)
    proc.wait()
    logfile.flush()
    errfile.flush()

def compile_and_get_time(args):
    logfile = open(args.stdout_log, 'w+')
    errfile = open(args.stderr_log, 'w+')
    outfile = open(args.stat_log + "_input_aware.csv", 'w+')
    outfile.write("graph,mode,inference_time,total_time\n")
    outfile.flush()

    output_path = args.out_path + "codegen/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_path + "build/"):
        os.makedirs(output_path + "build/")

    for mode in modes:
        for graph in graphs:
            curr = f">>>Running [{sp} sample size] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            if (mode == "schedule"):
                job_args = ['../../build/tests/gala_inference',
                            '../../tests/GALA-DSL/gcn/' + graph + '/h100.txt',
                            output_path]

                run(job_args, logfile, errfile)
            else:
                job_args = ['../../build/tests/gala_inference',
                            '../../tests/GALA-DSL/ablations/input-optimize/' + graph + '.txt',
                            output_path]

                run(job_args, logfile, errfile)

            job_args = ['make',
                        '-j56']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['./gala_model']
            outfile.write(graph + "," + modes + ",")
            outfile.flush()
            run_at(job_args, outfile, errfile, output_path + "build/")

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

    import pandas as pd
    import numpy as np
    import math
    import scipy
    from scipy import stats

    vals = {"Reddit" : 0, "Products" : 0}

    gala_df = pd.read_csv(args.stat_log + "_input_aware.csv")
    for index, row in gala_df.iterrows():
        if row['mode'] == "schedule":
            vals[row['graph']] = row['inference_time']
        else:
            print("Speedup of the input aware compilation for graph:", row['graph'], ", is:", row['inference_time'] / vals[row['graph']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info_graph_scale", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'wise', 'stat'], default="gala",
                        help="Task to generate Figures 16 to 17.")
    parser.add_argument("--train", action='store_true',
                        help="Train the model")
    parser.set_defaults(train=False)
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
