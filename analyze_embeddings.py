import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stats import *


"""
Script to calculate stats/plots for 3-emb experiment

"""



def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Experiment run #',required=True)
    return parser.parse_args()

def read_file(model,run):
    file_path = os.path.join(os.getcwd(),'viz','run_{}'.format(run),'{}_embedding'.format(model.upper()),'emb_results_recall.csv')
    df = pd.read_csv(file_path)
    return df

def calculate_stats(df,model,root_dir):

    # Mean
    mean_test_inp_dist = get_mean(col=df['Test-Gz Cosine'])
    mean_train_inp_dist = get_mean(col=df['Train-Gz Cosine'])

    # Variance
    var_test_inp_dist = get_var(col=df['Test-Gz Cosine'])
    var_train_inp_dist = get_var(col=df['Train-Gz Cosine'])

    print('{} Test-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_test_inp_dist,var_test_inp_dist))
    print('{} Train-Gz Cosine :: Mean= {}  Var = {}'.format(model.upper(),mean_train_inp_dist,var_train_inp_dist))

    # Histogram
    plot_hist(col=df['Test-Gz Cosine'],fname=os.path.join(root_dir,'{}_test_inp_hist.png'.format(model.upper())))
    plot_hist(col=df['Train-Gz Cosine'],fname=os.path.join(root_dir,'{}_train_inp_hist.png'.format(model.upper())))

def create_box_plot(df_list,mode='test',root_dir=None):

    """
    Takes a list of dataframes for different models
    Creates box plots for test/train -- G(z) cosine distance
    values

    """
    col_list = []
    if mode == 'test':
        for df in df_list:
            col_list.append(df['Test-Gz Cosine'])
    else:
        for df in df_list:
            col_list.append(df['Train-Gz Cosine'])


    df_concat = pd.concat(col_list,axis=1)
    df_concat.columns = models
    generate_box_plot(df_concat,fname=os.path.join(root_dir,'{}_box_plot.png'.format(mode)),mode=mode)

    # Homegenity tests
    if mode == 'test':
        print ('### Homogenity test for G(z) - Test Image distances ###')
    else:
        print('### Homogenity test for G(z) - Closest Training  Image distances ###')

    print('Homegenity tests for DCGAN/DCGAN-GP')
    check_homegenity(df_concat['dcgan'],df_concat['dcgan-gp'])
    print('Homegenity tests for DCGAN/DCGAN_SIM')
    check_homegenity(df_concat['dcgan'],df_concat['dcgan_sim'])
    print('Homegenity tests for WGAN/WGAN-GP')
    check_homegenity(df_concat['wgan'],df_concat['wgan-gp'])
    print('Homegenity tests for DCGAN/DRAGAN (With BN)')
    check_homegenity(df_concat['dcgan'],df_concat['dragan'])
    print('Homegenity tests for DCGAN/DRAGAN (No BN)')
    check_homegenity(df_concat['dcgan'],df_concat['dragan_bn'])
    print('Homegenity tests for DRAGAN /DRAGAN (With BN)')
    check_homegenity(df_concat['dragan'],df_concat['dragan_bn'])
    print('Homegenity tests for DCGAN/DCGAN-CONS')
    check_homegenity(df_concat['dcgan'],df_concat['dcgan-cons'])




if __name__ == '__main__':
    args = build_parser()
    root_dir = os.path.join(os.getcwd(),'viz','run_{}'.format(args.run),'embeddings')

    if os.path.exists(root_dir) is False:
        os.makedirs(root_dir)

    df_list = []

    for model in models:
        df = read_file(model,args.run)
        df_list.append(df)
        calculate_stats(df=df,model=model,root_dir=root_dir)

    create_box_plot(df_list,mode = 'test',root_dir=root_dir)
    create_box_plot(df_list,mode = 'train',root_dir=root_dir)

