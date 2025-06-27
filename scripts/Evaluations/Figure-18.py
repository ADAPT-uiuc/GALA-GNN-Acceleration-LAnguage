import argparse
import subprocess
import os

build_path = r"../../build/"

layers = ["2", "3", "4", "8"]
hidden_dims = ["32", "64", "128", "256", "512", "1024"]

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

    outfile = open(args.stat_log + "_scalability_GALA.csv", 'w+')
    outfile.write("layers,hidden,inference_time,total_time\n")
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

    for li in layers:
        for dim in hidden_dims:
            curr = f">>>Running [{li} layers with {dim} ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['../../build/tests/gala_inference',
                        '../../tests/GALA-DSL/ablations/scalability/' + li + '_' + dim + '.txt',
                        output_path]
            run(job_args, logfile, errfile)

            curr = f">>>Building code for [{li} layers with {dim} ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['cmake',
                        '-DCMAKE_PREFIX_PATH="/home/damitha2/new_torch/libtorch"',
                        '-DCAFFE2_USE_CUDNN=True',
                        '..']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['make',
                        '-j56']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['./gala_model']
            outfile.write(li + "," + dim + ",")
            outfile.flush()
            run_at(job_args, outfile, errfile, output_path + "build/")

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def evalWise(args):
    print("WiseGraph: Empty for now")

def createFigure(args):
    print("Create figure: Empty for now")

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "wise"):
        evalWise(args)
    elif (args.job == "fig"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info", help="File to store timing data")
    parser.add_argument("--job", type=str, choices=['gala', 'wise', 'fig'], default="gala",
                        help="Task to generate Figures 18.")
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
