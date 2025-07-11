import argparse
import subprocess
import os

build_path = r"../../build/"

dgl_map = {"papers100M":"ogbn-papers100M"}

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
    outfile = open(args.stat_log + "_node_sampling.csv", 'w+')
    outfile.write("sample,inference_time,total_time\n")
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

    for sp in sample_precen:
        curr = f">>>Running [{sp} sample size] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print(curr)
        logfile.write(curr+"\n")
        errfile.write(curr+"\n")

        job_args = ['../../build/tests/gala_inference_long',
                    '../../tests/GALA-DSL/ablations/scalability/graph_' + sp + '.txt',
                    output_path]

        run(job_args, logfile, errfile)

        job_args = ['cmake',
                    '-DCMAKE_PREFIX_PATH="/home/damitha2/new_torch/libtorch"',
                    '-DCAFFE2_USE_CUDNN=True',
                    '..']
        run_at(job_args, logfile, errfile, output_path + "build/")
        job_args = ['make',
                    '-j56']
        run_at(job_args, logfile, errfile, output_path + "build/")
        job_args = ['./gala_model']
        outfile.write(sp + ",")
        outfile.flush()
        run_at(job_args, outfile, errfile, output_path + "build/")

        logfile.write(("<"*100)+"\n")
        errfile.write(("<"*100)+"\n")
        print("<"*100)
    logfile.close()
    errfile.close()

def evalDGL(args):
    dgl_working_path = r"../../tests/Baselines/DGL"

    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_DGL_node_sampling.csv", 'w+')
    outfile.write("sample,inference_time,total_time\n")
    outfile.close()

    for sp in sample_precen:
        curr = f">>>Running [{sp} sample size] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print(curr)
        logfile.write(curr+"\n")
        errfile.write(curr+"\n")

        job_args = ['python',
                    '../../tests/Baselines/DGL/benchmark_dgl_gcn_node_sampling.py',
                    '--dataset', "ogbn-papers100M",
                    '--n-hidden', str(32),
                    '--pi', sp,
                    '--layers', str(1),
                    '--n-epochs', str(100),
                    "--logfile", args.stat_log + "_" + args.hw + "_DGL_node_sampling.csv",
                    "--device", "cuda",
                    "--skip_train",
                    "--discard", str(5)]
        outfile = open(args.stat_log + "_" + args.hw + "_DGL_node_sampling.csv", 'a+')
        outfile.write(sp + ",")
        outfile.close()
        run(job_args, logfile, errfile)

        logfile.write(("<"*100)+"\n")
        errfile.write(("<"*100)+"\n")
        print("<"*100)
    logfile.close()
    errfile.close()

def createFigure(args):
    import numpy as np
    import pandas as pd

    vals_dgl = {}
    vals_wise = {}
    for sp in sample_precen:
        vals_dgl[int(sp)] = 0
        vals_wise[int(sp)] = 0

    wise_df = pd.read_csv("results_table5.csv")
    for index, row in wise_df.iterrows():
        vals_wise[int(row['dataset'].split("_")[-1])] = row['inference_time']
    dgl_df = pd.read_csv(args.stat_log + "_DGL_node_sampling.csv")
    for index, row in dgl_df.iterrows():
        if (row['inference_time'] != np.nan):
            vals_dgl[int(row['sample'])] = row['inference_time']
    gala_df = pd.read_csv(args.stat_log + "_node_sampling.csv")
    for index, row in gala_df.iterrows():
        if (vals_dgl[int(row['sample'])] == 0 or pd.isna(vals_dgl[int(row['sample'])])):
            dgl_val = "OOM"
        else:
            dgl_val = "{:.2f}".format(vals_dgl[int(row['sample'])]*1000) 

        if (vals_wise[int(row['sample'])] == 0 or vals_wise[int(row['sample'])] == np.nan):
            wise_val = "OOM"
        else:
            wise_val = "{:.2f}".format(vals_wise[int(row['sample'])]*1000) 
        print("Node sample percentage:", int(row['sample']),"GALA:", "{:.2f}".format(row['inference_time']*1000) ,"DGL:", dgl_val,"WiseGraph:", wise_val)

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "dgl"):
        evalDGL(args)
    elif (args.job == "stat"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info_graph_scale", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'stat'], default="gala",
                        help="Task to generate Table 5.")
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
