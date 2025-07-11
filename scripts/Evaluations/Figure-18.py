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


def createFigure(args):
    import pandas as pd
    vals_wise = {}
    vals_gala = {}
    for l in layers:
        vals_wise[int(l)] = {}
        vals_gala[int(l)] = {}
        for dim in hidden_dims:
            vals_wise[int(l)][int(dim)] = 0
            vals_gala[int(l)][int(dim)] = 0

    wise_df = pd.read_csv("results_fig18_19.csv")
    for index, row in wise_df.iterrows():
        if row['dataset'] == "reddit":
            vals_wise[row['num_layer']][row['hidden_feat']] = row['inference_time']

    gala_df = pd.read_csv(args.stat_log + "_scalability_GALA.csv")
    for index, row in gala_df.iterrows():
        print("Layers:", int(row['layers']),"Hidden dim:",int(row['hidden']),"Speedup:", vals_wise[row['layers']][row['hidden']] / row['inference_time'])
        vals_gala[row['layers']][row['hidden']] = row['inference_time']

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    row_labels = [li for li in vals_gala]
    col_labels = [hi for hi in vals_gala[2]]
    data = np.zeros(shape=(len(row_labels), len(col_labels)))

    li_i = 0
    for li in vals_gala:
        hi_i = 0
        for hi in vals_gala[li]:
            data[li_i, hi_i] = vals_wise[li][hi] / vals_gala[li][hi]
            hi_i += 1
        li_i += 1
    df = pd.DataFrame(data, index=row_labels, columns=col_labels)
    
    # Create the heatmap
    plt.figure(figsize=(6, 3))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    
    # Add a title
    # plt.title(g)
    plt.ylabel("Layers")
    plt.xlabel("Hidden dimensions")
    plt.savefig("Figure 18.pdf", format="pdf", bbox_inches="tight", pad_inches=0, dpi=1000)

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "stat"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info", help="File to store timing data")
    parser.add_argument("--job", type=str, choices=['gala', 'wise', 'stat'], default="gala",
                        help="Task to generate Figures 18.")
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
