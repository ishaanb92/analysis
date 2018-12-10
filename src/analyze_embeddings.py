import pandas as pd
import numpy as np
import os,sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from stats import *
import math
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
"""
Script to calculate stats/plots for 3-emb experiment

"""



def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Experiment run #',required=True)
    parser.add_argument('--dataset',type=str,help='Dataset to analyze',default='mnist',required=False)
    return parser.parse_args()

def read_csv_file(model,run,dataset='celeba'):
    file_path = os.path.join(os.getcwd(),'results',dataset,'run_{}'.format(run),'three_vec',model.lower(),'emb_results_recall.csv')
    df = pd.read_csv(file_path)
    return df

def concat_df(run,dataset='celeba'):
    """
    Reads all CSV files containing results of the 3-vec experiment.
    Creates 3 dataframes, one containing all the 'Test' distances,
    one containing 'Train' distances and one for the 'Gaps'

    """
    test_col_list = []
    train_col_list = []
    gap_col_list = []

    for model in models:
        df = read_csv_file(model,run,dataset)
        test_col_list.append(df['Test-Gz Cosine'])
        train_col_list.append(df['Train-Gz Cosine'])
        gap_col_list.append(df['Test-Gz Cosine'].sub(df['Train-Gz Cosine']))


    df_test = pd.concat(test_col_list,axis=1)
    df_test.columns = models

    df_train = pd.concat(train_col_list,axis=1)
    df_train.columns = models

    df_gap = pd.concat(gap_col_list,axis=1)
    df_gap.columns = models



    return df_test,df_train,df_gap



def calculate_embedding_stats(df):

    dist_means = {}
    dist_vars = {}

    # Mean + Variance
    for model in models:
        mean = df[model].mean()
        dist_means[model] = mean
        var = df[model].var()
        dist_vars[model] = var

    return dist_means,dist_vars



def create_box_plot(df,mode='test',out_dir=None):

    """
    Takes a dataframe containing the train or test distances (depending on the mode)
    Creates box plots for test/train -- G(z) cosine distance values

    """

    kwds = {}
    kwds['patch_artist'] = True
    generate_box_plot(df,fname=os.path.join(out_dir,'{}_box_plot.png'.format(mode)),mode=mode,kwds=kwds)


def create_histogram(col,model,out_dir,mode='test'):
    fname = os.path.join(out_dir,'{}_{}_inp_hist.png'.format(model.upper(),str(mode)))
    plot_hist(col=col,fname=fname)


def analyze_embeddings(run,draw=False,log_file=None,dataset='mnist'):

    out_dir = os.path.join(os.getcwd(),'figures',dataset,'run_{}'.format(run),'embeddings')

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    # Code to plot embeddings for MNIST
   # if dataset == 'mnist':
   #     fname = os.path.join(out_dir,'mnist_plot_dict.pkl')
   #     with open(fname,'rb') as f:
   #         try:
   #             mnist_plot_dict = pickle.load(f)
   #         except:
   #             print('Error reading file,exiting')
   #             return None,None,None
   #     colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff','#ff00ff', '#990000', '#999900', '#009900', '#009999']
   #     groups = [str(i) for i in range(10)]
   #     data = []
   #     for number in groups:
   #         data.append(mnist_plot_dict[int(number)])

   #     data = tuple(data)
   #     # Create plot
   #     fig = plt.figure()

   #     for embs,color,group in zip(data,colors,groups):
   #         x= []
   #         y = []

   #         for emb in embs:
   #             x.append(emb[0])
   #             y.append(emb[1])

   #         plt.plot(x, y,'.',c=color,label=group)

   #     plt.title('MNIST embeddings using Center Loss')
   #     plt.legend()
   #     fname_plot = os.path.join(out_dir,'mnist_embs_plot.png')
   #     #TODO : Fix legend placement
   #     plt.savefig(fname=fname_plot)

        #TODO : Plot (closest) train/ test / G(z) images for all GANs



    df_test,df_train,df_gap = concat_df(run,dataset)

    dist_means_test,dist_var_test = calculate_embedding_stats(df_test)
    dist_means_train,dist_var_train = calculate_embedding_stats(df_train)

    for model in models:
        mean_gap = math.fabs(dist_means_test[model]-dist_means_train[model])
        if log_file is not None:
            log_file.write('{} :: test-gz mean distance : {} train-gz distance : {} mean gap : {} \n'.format(model.upper(),dist_means_test[model],dist_means_train[model],mean_gap))
        else:
            print('{} :: test-gz mean distance : {} train-gz distance : {} mean gap : {}'.format(model.upper(),dist_means_test[model],dist_means_train[model],mean_gap))

    #Homogenity Checks
    pairs = generate_pairs()

    for pair in pairs:
        if check_homegenity(df_test[pair[0]],df_test[pair[1]]) is True:
            if log_file is not None:
                log_file.write('Test-Gz distances computed for models {} and {} are homogenous\n'.format(pair[0].upper(),pair[1].upper()))
            else:
                print('Test-Gz distances computed for models {} and {} are homogenous'.format(pair[0].upper(),pair[1].upper()))

        if check_homegenity(df_train[pair[0]],df_train[pair[1]]) is True:
            if log_file is not None:
                log_file.write('Train-Gz distances computed for models {} and {} are homogenous\n'.format(pair[0].upper(),pair[1].upper()))
            else:
                print('Train-Gz distances computed for models {} and {} are homogenous'.format(pair[0].upper(),pair[1].upper()))


    if draw is True:
        create_box_plot(df=df_test,mode='test',out_dir=out_dir)
        create_box_plot(df=df_train,mode='train',out_dir=out_dir)
        create_box_plot(df=df_gap,mode='gap',out_dir=out_dir)
        for model in models:
            create_histogram(col=df_test[model],model=model,out_dir=out_dir,mode='test')
            create_histogram(col=df_train[model],model=model,out_dir=out_dir,mode='train')



        return dist_means_test,dist_means_train,dist_var_test,dist_var_train

if __name__ == '__main__':
    args = build_parser()
    _,_,_,_ = analyze_embeddings(run=args.run,draw=True)


