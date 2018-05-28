import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *


"""
Script to calculate stats/plots for metric results

"""


def calculate_stats(df,root_dir):

    # Mean/Var
    for model in models:
        print('{} :: Mean = {} Var = {}'.format(model.upper(),get_mean(df[model.upper()]),get_var(df[model.upper()])))
        plot_hist(col=df[model.upper()],fname=os.path.join(root_dir,'{}_metric_hist.png'.format(model)))

    # Box-Plot
    valid_cols = []  # Weird pandas bug
    for model in models:
        valid_cols.append(df[model.upper()])

    df_concat = pd.concat(valid_cols,axis=1)

    generate_box_plot(df=df_concat,fname=os.path.join(root_dir,'box_plot.png'))

    # Homegenity Tests
    print('Homegenity tests for DCGAN/DCGAN-GP')
    check_homegenity(df['DCGAN'],df['DCGAN-GP'])
    print('Homegenity tests for WGAN/WGAN-GP')
    check_homegenity(df['WGAN'],df['WGAN-GP'])
    print('Homegenity tests for DCGAN/DRAGAN')
    check_homegenity(df['DCGAN'],df['DRAGAN'])
    print('Homegenity tests for DCGAN/DCGAN-CONS')
    check_homegenity(df['DCGAN'],df['DCGAN-CONS'])




if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(),'viz','metric')
    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    df = pd.read_csv('/home/fungii/thesis_code/celebA_metric_results/gan_distances.csv')
    calculate_stats(df=df,root_dir=root_dir)

