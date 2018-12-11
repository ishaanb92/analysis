import numpy as np
import os,sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from stats import *
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset',type=str,help='mnist/celeba',required=True)
    parser.add_argument('--run',type=int,help='Index of run',required=True)
    return parser



def plot_embeddings(dataset='mnist',run=1):
    """
    General function to create a scatter-plot of
    2-D image embeddings. It is assumed that either
    the embeddings are 2-D or have been reduced to 2-D
    using standard dimensionality reduction techniques

    """

    if dataset == 'mnist':
        #Read the dictionary
        dict_dir = os.path.join(os.getcwd(),'results',dataset,'run_{}'.format(run),'embeddings')
        try:
            with open(os.path.join(dict_dir,'mnist_plot_dict.pkl'),'rb') as f:
                plot_dict = pickle.load(f)

        except Exception as ex:
            print(ex)

        # Plotting
        colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff','#ff00ff', '#990000', '#999900', '#009900', '#009999']
        groups = [str(i) for i in range(10)]
        data = []
        for number in groups:
            data.append(plot_dict[int(number)])

        data = tuple(data)
        # Create plot
        fig = plt.figure(figsize=(8,5))

        for embs,color,group in zip(data,colors,groups):
            x= []
            y = []
            for emb in embs:
                x.append(emb[0])
                y.append(emb[1])
            plt.plot(x, y,'.',c=color,label=group)

        plt.title('MNIST embeddings using Center Loss')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=10)
        out_dir = os.path.join(os.getcwd(),'figures',dataset,'run_{}'.format(run),'embeddings')
        fname_plot = os.path.join(out_dir,'mnist_embs_plot.png')
        plt.savefig(fname=fname_plot)

    else:
        print('Other datasets not supported, will do if need arises')



if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    plot_embeddings(dataset=args.dataset,run=args.run)
