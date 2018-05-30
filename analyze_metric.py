import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *


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
    for model in models:
        print('{} :: Mean = {} Var = {}'.format(model.upper(),get_mean(df[model.upper()]),get_var(df[model.upper()])))
        plot_hist(col=df[model.upper()],fname=os.path.join(root_dir,'{}_metric_hist.png'.format(model.upper())))


    generate_box_plot(df=df_concat,fname=os.path.join(root_dir,'box_plot.png'))

    # Homegenity Tests
    print('Homegenity tests for DCGAN/DCGAN-GP')
    check_homegenity(df_concat['DCGAN'],df_concat['DCGAN-GP'])
    print('Homegenity tests for WGAN/WGAN-GP')
    check_homegenity(df_concat['WGAN'],df_concat['WGAN-GP'])
    print('Homegenity tests for DCGAN/DRAGAN')
    check_homegenity(df_concat['DCGAN'],df_concat['DRAGAN'])
    print('Homegenity tests for DCGAN/DRAGAN (No BN)')
    check_homegenity(df_concat['DCGAN'],df_concat['DRAGAN_NO_BN'])
    print('Homegenity tests for DRAGAN/DRAGAN (No BN)')
    check_homegenity(df_concat['DRAGAN'],df_concat['DRAGAN_NO_BN'])
    print('Homegenity tests for DCGAN/DCGAN-CONS')
    check_homegenity(df_concat['DCGAN'],df_concat['DCGAN-CONS'])




if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(),'viz','metric')
    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    # Merge BN and non-BN data-frames (for DRAGAN column)
    df = pd.read_csv('/home/fungii/thesis_code/celebA_metric_results/gan_distances.csv')
    df_no_bn = pd.read_csv('/home/fungii/thesis_code/celebA_metric_results/gan_distances_sanity.csv')
    df_rename = df_no_bn.rename(columns={"DRAGAN":"DRAGAN_NO_BN"})
    df_concat = pd.concat([df,df_rename["DRAGAN_NO_BN"]],axis=1)

    calculate_stats(df=df_concat,root_dir=root_dir)

