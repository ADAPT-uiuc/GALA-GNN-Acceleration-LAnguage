import argparse
import subprocess
import os

build_path = r"../../build/"

datasets = ["Reddit", "Products"]
samples = ["data", "dynamic", "kernel", "no-sampling"]

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
    outfile = open(args.stat_log + "_sampling.csv", 'w+')
    outfile.write("dataset,sampling,inference_time,accuracy\n")
    outfile.flush()

    # TODO add build
    if not os.path.exists(build_path):
        os.makedirs(build_path)
        job_args = ['cmake',
                    '..']
        run_at(job_args, logfile, errfile, build_path)
        job_args = ['make',
                    '-j56']
        run_at(job_args, logfile, errfile, build_path)

    output_path = args.out_path + "codegen/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_path + "build/"):
        os.makedirs(output_path + "build/")

    for dst in datasets:
        for sm in samples:
            curr = f">>>Running [{sm} sample size] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['../../build/tests/gala_inference_sample',
                        '../../tests/GALA-DSL/ablations/sampling/' + sm + '/' + dst + '.txt',
                        output_path]

            run(job_args, logfile, errfile)

            job_args = ['make',
                        '-j56']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['./gala_model']
            outfile.write(dst + "," + sm + ",")
            outfile.flush()
            run_at(job_args, outfile, errfile, output_path + "build/")

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def createFigure(args):
    import pandas as pd
    import numpy as np
    import math
    import scipy
    from scipy import stats

    gala_df = pd.read_csv(args.stat_log + "_sampling.csv")
    for index, row in gala_df.iterrows():
        print("Method:",row['sampling'],"Dataset:",row['dataset'],"Runtime:",row['inference_time']*1000,"Accuracy:",row['accuracy'])

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "stat"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info_graph", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'stat'], default="gala",
                        help="Task to generate Table 6.")
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
