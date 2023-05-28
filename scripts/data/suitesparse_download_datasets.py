#!/usr/bin/python3
# from scripts.dataset_exp_utils import create_out_dir
from time import sleep
import pandas as pd
import suitesparse_dataset_exp_utils as du
import sys
import datetime as dt
import os as os

from signal import signal, SIGINT


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)


if __name__ == '__main__':
    signal(SIGINT, handler)

    date = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    logfile = open("log_"+date+"", "w")

    options, args = du.parse_args(sys.argv[1:])
    print("Arguments passed to the script: "+str(options), file=logfile)
    # print("Outdir: "+options['dir'])
    outdir = du.create_out_dir(options)

    matrices = du.get_matrices(options)
    print("Matrices processed: "+str(matrices['Name'].to_numpy()), file=logfile)

    num_mtx = matrices.shape[0]
    exp_index = 0
    for index, row in matrices.iterrows():
        print("Processing ["+str(exp_index)+"/"+str(num_mtx)+"] mtx: "+row['Name'], file=sys.stdout)
        print("Processing ["+str(exp_index)+"/"+str(num_mtx)+"]", file=logfile)
            
        du.download_matrix(group=row['Group'], name=row['Name'], directory=outdir, logfile=logfile)
        
        sleep(5)
        
        exp_index = exp_index+1
        du.remove_matrix_aux_files(group=row['Group'], name=row['Name'], directory=outdir, logfile=logfile)

    logfile.close()
