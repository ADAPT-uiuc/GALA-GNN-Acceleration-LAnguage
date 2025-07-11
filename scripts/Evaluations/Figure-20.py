import argparse
import subprocess
import os

build_path = r"../../build/"

graphs = ["Reddit", "Products"]
modes = ["schedule", "input-aware"]

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
            curr = f">>>Running [{mode} {graph}] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
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
            outfile.write(graph + "," + mode + ",")
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

    import matplotlib.pyplot as plt
    import seaborn as sns

    vals = {"Reddit" : 0, "Products" : 0}

    data_hand = {"Reddit" : 0, "Products" : 0}
    data_auto = {"Reddit" : 0, "Products" : 0}

    gala_df = pd.read_csv(args.stat_log + "_input_aware.csv")
    for index, row in gala_df.iterrows():
        if row['mode'] == "schedule":
            vals[row['graph']] = row['inference_time']
            data_hand[row['graph']] = row['inference_time']
        else:
            print("Speedup of the input aware compilation for graph:", row['graph'], ", is:", vals[row['graph']] / row['inference_time'])
            data_auto[row['graph']] = row['inference_time']

    df_data = []
    df_cols = ["x", "Schedule selection", "Runtime"]

    graph_tune = ['Reddit', 'Products']

    for g in graph_tune:
        if g == 'Products':
            df_data.append(['OGBN-\nProducts',"Hand-select", data_hand[g]])
            df_data.append(['OGBN-\nProducts',"Input-aware", data_auto[g]])
        else:
            df_data.append([g,"Hand-select", data_hand[g]])
            df_data.append([g,"Input-aware", data_auto[g]])

    # df = pd.DataFrame(data['GCN'], columns=["DGL","GALA"])
    df = pd.DataFrame(df_data, columns=df_cols)

    custom_colors = [
        '#E377C2',  # Dark Magenta
        '#FFCC99',  # Light Peach
    ]

    h = 1.5
    w = 5
    plt.figure(figsize=(w, h))
        
    # Create a bar plot
    ax = sns.barplot(y='x', x='Runtime', hue=df["Schedule selection"], data=df, orient='h', palette=custom_colors)
    ax.xaxis.set_tick_params(pad=0)

    hatches = ['', '.']
    edge_color = ['white', 'black']

    for i, bar in enumerate(ax.patches):
        if i < 4:
            bar.set_hatch(hatches[i // 2])
            bar.set_edgecolor(edge_color[i // 2])
        else:
            bar.set_hatch(hatches[i % 2])
            bar.set_edgecolor(edge_color[i % 2])

    # ax.legend(loc='upper right', title="Schedule selection", ncol=2, fontsize=14, borderaxespad=0.1, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)
    leg = plt.legend(loc=(-0.1,1), ncol=5, fontsize=18, frameon=False, borderaxespad=0.1, columnspacing=0.5, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)

    plt.xlabel('Execution Time(ms)', labelpad=0)
    plt.ylabel('')

    plt.savefig("Figure 20.pdf", format="pdf", bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.show()


def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "stat"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info_graph_scale", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'stat'], default="gala",
                        help="Task to generate Figure 20.")
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
