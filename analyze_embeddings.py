import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *




def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model',type=str,help='GAN model : DCGAN/WGAN/...',required=True)
    return parser

def read_file(model):
    file_path = os.path.join(os.getcwd(),'viz','{}_embedding'.format(model.upper()),'emb_results_recall.csv')
    df = pd.read_csv(file_path)
    return df

def generate_stats(df,model):

    # Mean
    mean_test_inp_dist = get_mean(col=df['Test-Gz Cosine'])
    mean_train_inp_dist = get_mean(col=df['Train-Gz Cosine'])

    # Variance
    var_test_inp_dist = get_var(col=df['Test-Gz Cosine'])
    var_train_inp_dist = get_var(col=df['Train-Gz Cosine'])

    print('{} Test-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_test_inp_dist,var_test_inp_dist))
    print('{} Train-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_train_inp_dist,var_train_inp_dist))

    # Histogram
    plot_hist(col=df['Test-Gz Cosine'],model=model,fname='test_inp_hist')
    plot_hist(col=df['Train-Gz Cosine'],model=model,fname='train_inp_hist')

def create_box_plot(df_list,mode='test'):

    """
    Takes a list of dataframes for different models
    Creates box plots for test/train -- G(z) cosine distance
    values

    """
    col_list = []
    for df in df_list:
        col_list.append(df['Test-Gz Cosine'])

    df_concat = pd.concat(col_list,axis=1)
    df_concat.columns = models
    ax = generate_box_plot(df_concat)
    plt.show()



if __name__ == '__main__':
    df_list = []
    for model in models:
        df = read_file(model)
        df_list.append(df)
        generate_stats(df=df,model=model)

    create_box_plot(df_list)
