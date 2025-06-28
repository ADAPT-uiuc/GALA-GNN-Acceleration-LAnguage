import argparse
import subprocess
import os

build_path = r"../../build/"

exec_types = ["memory", "time"]
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
    outfile = open(args.stat_log + "_memory.csv", 'w+')
    outfile.write("exec,memory,total_time\n")
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

    for exec in exec_types:
        curr = f">>>Running [Execute {exec} to get time and memory consumption] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print(curr)
        logfile.write(curr+"\n")
        errfile.write(curr+"\n")

        job_args = ['../../build/tests/gala_train',
                    '../../tests/GALA-DSL/ablations/memory-consumption/' + exec +'.txt',
                    output_path]
        run(job_args, logfile, errfile)

        job_args = ['make',
                    '-j56']
        run_at(job_args, logfile, errfile, output_path + "build/")
        job_args = ['./gala_model']
        outfile.write(exec + ",")
        outfile.flush()
        run_at(job_args, outfile, errfile, output_path + "build/")

        logfile.write(("<"*100)+"\n")
        errfile.write(("<"*100)+"\n")
        print("<"*100)
    logfile.close()
    errfile.close()

def evalDGL(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    curr = f">>>Testing [{dset} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    print(curr)
    logfile.write(curr+"\n")
    errfile.write(curr+"\n")

    job_args = ['python',
                '../../tests/Baselines/DGL/benchmark_dgl_'+model+'_memory.py',
                '--dataset', dgl_map[dset],
                '--n-hidden', str(32),
                '--layers', str(1),
                '--n-epochs', str(100),
                "--logfile", args.stat_log + "_memory.csv",
                "--device", "cuda",
                "--discard", str(5)]
    outfile = open(args.stat_log + "_memory.csv", 'w+')
    outfile.write("dgl,")
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
    elif (args.job == "fig"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info", help="File to store timing data")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'wise', 'fig'], default="gala",
                        help="Task to generate Figures 16 to 17.")
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
