import pandas as pd
import numpy as np
import os
from stats import *
from argparse import ArgumentParser


"""
Script to calculate stats/plots for metric results

"""


def calculate_metric_stats(df,root_dir,draw=False,log_file=None,dataset='celeba'):

    valid_cols = []  # Weird pandas bug -- First column is garbage -- remove it
                     # 2nd column is the original image file path, not needed for numerical analysis
    for model in models:
        valid_cols.append(df[model.upper()])

    df_concat = pd.concat(valid_cols,axis=1)

    mean_dict = {}

    # Mean/Var
    for col in df_concat.columns:
        mean_dict[col.lower()] = get_mean(df[col])
        if log_file is not None:
            log_file.write('{} :: Mean = {} Var = {}\n'.format(col,mean_dict[col.lower()],get_var(df[col])))
        else:
            print('{} :: Mean = {} Var = {}'.format(col,mean_dict[col.lower()],get_var(df[col])))

        plot_hist(col=df[col],fname=os.path.join(root_dir,'{}_metric_hist.png'.format(col)),metric=True)

    kwds = {}
    kwds['patch_artist'] = True
    if draw is True:
        generate_box_plot(df=df_concat,fname=os.path.join(root_dir,'box_plot.png'),kwds=kwds)

    # Homogenity tests
    pairs = generate_pairs()

    for pair in pairs:
        if check_homegenity(df_concat[pair[0].upper()],df_concat[pair[1].upper()]) is True:
            if log_file is not None:
                log_file.write('Distances computed for models {} and {} are homogenous\n'.format(pair[0].upper(),pair[1].upper()))
            else:
                print('Distances computed for models {} and {} are homogenous'.format(pair[0].upper(),pair[1].upper()))

    return mean_dict


def analyze_metric(run,draw=False,log_file=None,dataset='celeba'):

    root_dir = os.path.join(os.getcwd(),'viz',dataset,'run_{}'.format(run),'metric')
    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    df = pd.read_csv(os.path.join(os.getcwd(),'metric_results',dataset,'run_{}'.format(run),'gan_distances_{}.csv'.format(dataset)))

    return calculate_metric_stats(df=df,root_dir=root_dir,draw=draw,log_file=log_file,dataset=dataset)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Run x of experiment',default='1')
    parser.add_argument('--dataset',type=str,help='celeba/mnist',default='celeba')
    args = parser.parse_args()

    analyze_metric(run=args.run,draw=True,dataset=args.dataset)


