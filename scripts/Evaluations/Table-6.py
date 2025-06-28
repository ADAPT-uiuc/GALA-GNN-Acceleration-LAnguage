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
    outfile = open(args.stat_log + "_graph.csv", 'w+')
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

        job_args = ['../../build/tests/gala_inference',
                    '../../tests/GALA-DSL/ablations/scalability/graph_' + sp + + '.txt',
                    output_path]

        run(job_args, logfile, errfile)

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

    percen = [1, 2, 5, 10, 20]

    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_DGL_node_sampling.csv", 'w+')
    outfile.write("sample,inference_time,total_time\n")
    outfile.close()

    for pi in percen:
        curr = f">>>Testing [{pi} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print(curr)
        logfile.write(curr+"\n")
        errfile.write(curr+"\n")

        job_args = ['python',
                    '../../tests/Baselines/DGL/benchmark_dgl_gcn_node_sampling.py',
                    '--dataset', "ogbn-papers100M",
                    '--n-hidden', str(32),
                    '--pi', str(pi),
                    '--layers', str(1),
                    '--n-epochs', str(100),
                    "--logfile", args.stat_log + "_" + args.hw + "_DGL_node_sampling.csv",
                    "--device", "cuda",
                    "--skip_train",
                    "--discard", str(5)]
        outfile = open(args.stat_log + "_" + args.hw + "_DGL_node_sampling.csv", 'a+')
        outfile.write(pi + ",")
        outfile.close()
        run(job_args, logfile, errfile)

        logfile.write(("<"*100)+"\n")
        errfile.write(("<"*100)+"\n")
        print("<"*100)
    logfile.close()
    errfile.close()

def evalWise(args):
    print("WiseGraph: Empty for now")

def createFigure(args):
    print("Create Figure 16-17: Empty for now")

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "dgl"):
        evalDGL(args)
    elif (args.job == "wise"):
        evalWise(args)
    elif (args.job == "stat"):
        createFigure(args)

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
