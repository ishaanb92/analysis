import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *
from argparse import ArgumentParser


"""
Script to calculate stats/plots for metric results

"""


def calculate_stats(df,root_dir):

    valid_cols = []  # Weird pandas bug -- First column is garbage -- remove it
                     # 2nd column is the original image file path, not needed for numerical analysis
    for idx in range(2,len(df.columns)):
        valid_cols.append(df[df.columns[idx]])

    df_concat = pd.concat(valid_cols,axis=1)

    # Mean/Var
    for col in df_concat.columns:
        print('{} :: Mean = {} Var = {}'.format(col,get_mean(df[col]),get_var(df[col])))
        plot_hist(col=df[col],fname=os.path.join(root_dir,'{}_metric_hist.png'.format(col)))


    generate_box_plot(df=df_concat,fname=os.path.join(root_dir,'box_plot.png'))

    # Homegenity Tests
    print('Homegenity tests for DCGAN/DCGAN-GP')
    check_homegenity(df_concat['DCGAN'],df_concat['DCGAN-GP'])
    print('Homegenity tests for WGAN/WGAN-GP')
    check_homegenity(df_concat['WGAN'],df_concat['WGAN-GP'])
    print('Homegenity tests for DCGAN/DRAGAN')
    check_homegenity(df_concat['DCGAN'],df_concat['DRAGAN'])
    print('Homegenity tests for DCGAN/DRAGAN (With BN)')
    check_homegenity(df_concat['DCGAN'],df_concat['DRAGAN_BN'])
    print('Homegenity tests for DRAGAN/DRAGAN (With BN)')
    check_homegenity(df_concat['DRAGAN'],df_concat['DRAGAN_BN'])
    print('Homegenity tests for DCGAN/DCGAN-CONS')
    check_homegenity(df_concat['DCGAN'],df_concat['DCGAN-CONS'])
    print('Homegenity tests for DCGAN/DCGAN-SIM')
    check_homegenity(df_concat['DCGAN'],df_concat['DCGAN_SIM'])




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Run x of experiment',default='1')
    args = parser.parse_args()

    root_dir = os.path.join(os.getcwd(),'viz','run_{}'.format(str(args.run)),'metric')
    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    # Merge BN and non-BN data-frames (for DRAGAN column)
    df = pd.read_csv(os.path.join('/home/fungii/thesis_code/celebA_metric_results','run_{}'.format(str(args.run)),'gan_distances.csv'))
    calculate_stats(df=df,root_dir=root_dir)

