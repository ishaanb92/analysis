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

def generate_stats(df,model,indx):

    # Mean
    mean_test_inp_dist = get_mean(col=df['Test-Gz Cosine'])
    mean_train_inp_dist = get_mean(col=df['Train-Gz Cosine'])

    # Variance
    var_test_inp_dist = get_var(col=df['Test-Gz Cosine'])
    var_train_inp_dist = get_var(col=df['Train-Gz Cosine'])

    print('{} Test-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_test_inp_dist,var_test_inp_dist))
    print('{} Train-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_train_inp_dist,var_train_inp_dist))

    # Histogram
    plot_hist(col=df['Test-Gz Cosine'],model=model,fname='test_inp')
    plot_hist(col=df['Train-Gz Cosine'],model=model,fname='train_inp')



if __name__ == '__main__':
    indx = 0
    for model in models:
        df = read_file(model)
        generate_stats(df=df,model=model,indx=indx)
        indx+=2
