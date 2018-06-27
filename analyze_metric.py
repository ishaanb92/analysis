import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *
from argparse import ArgumentParser


"""
Script to calculate stats/plots for metric results

"""


def calculate_metric_stats(df,root_dir,draw=False,log_file=None):

    valid_cols = []  # Weird pandas bug -- First column is garbage -- remove it
                     # 2nd column is the original image file path, not needed for numerical analysis
    for idx in range(2,len(df.columns)):
        valid_cols.append(df[df.columns[idx]])

    df_concat = pd.concat(valid_cols,axis=1)

    mean_dict = {}

    # Mean/Var
    for col in df_concat.columns:
        mean_dict[col] = get_mean(df[col])
        if log_file is not None:
            log_file.write('{} :: Mean = {} Var = {}'.format(col,mean_dict[col],get_var(df[col])))
        else:
            print('{} :: Mean = {} Var = {}'.format(col,mean_dict[col],get_var(df[col])))

        plot_hist(col=df[col],fname=os.path.join(root_dir,'{}_metric_hist.png'.format(col)))


    if draw is True:
        generate_box_plot(df=df_concat,fname=os.path.join(root_dir,'box_plot.png'))

    # Homogenity tests
    pairs = generate_pairs()

    for pair in pairs:
        if check_homegenity(df_concat[pair[0].upper()],df_concat[pair[1].upper()]) is True:
            if log_file is not None:
                log_file.write('Distances computed for models {} and {} are homogenous'.format(pair[0].upper(),pair[1].upper()))
            else:
                print('Distances computed for models {} and {} are homogenous'.format(pair[0].upper(),pair[1].upper()))

    return mean_dict





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Run x of experiment',default='1')
    args = parser.parse_args()

    root_dir = os.path.join(os.getcwd(),'viz','run_{}'.format(str(args.run)),'metric')
    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    # Merge BN and non-BN data-frames (for DRAGAN column)
    df = pd.read_csv(os.path.join('/home/fungii/thesis_code/celebA_metric_results','run_{}'.format(str(args.run)),'gan_distances.csv'))
    calculate_metric_stats(df=df,root_dir=root_dir)

