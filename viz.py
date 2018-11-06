import numpy as np
import hypertools as hyp
import matplotlib.pyplot as plt
import pickle
import os
from argparse import ArgumentParser
from sklearn.decomposition import PCA
"""

Script to create t-SNE viz for embeddings

"""

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model',type=str,help='GAN model used to select the right dictionary',required=True)
    parser.add_argument('--reduce',type=str,help='GAN model used to select the right dictionary',default='pca')
    parser.add_argument('--dataset',type=str,help='mnist/celeba',default='celeba')
    parser.add_argument('--run',type=str,help='Experiment Run',default='1')
    return parser.parse_args()

def load_embeddings(args):

    emb_dir =  os.path.join(os.getcwd(),'viz',args.dataset,'run_{}'.format(args.run),'embeddings')

    # Load train
    with open(os.path.join(os.getcwd(),'viz',args.dataset,'run_{}'.format(args.run),'{}_{}_embedding'.format(args.model.upper(),args.dataset.upper()),'closest_train_emb.pkl'),'rb') as f:
        train_dict = pickle.load(f)

    # Load test
    with open(os.path.join(emb_dir,'test_{}_emb_dict.pkl'.format(args.dataset.lower())),'rb') as f:
        test_dict = pickle.load(f)

    # Load G(z) -- specific to model
    with open(os.path.join(emb_dir,'{}_{}_emb_dict.pkl'.format(args.model.lower(),args.dataset.lower())),'rb') as f:
        gz_dict = pickle.load(f)

    return train_dict,test_dict,gz_dict

def add_embeddings(matrix,labels,emb_dict,hue):
    for path,emb in emb_dict.items():
        matrix.append(emb)
        labels.append(str(hue))
    return matrix,labels


def create_matrix(args):

    train_dict,test_dict,gz_dict =  load_embeddings(args)
    matrix = []
    labels = []

    matrix,labels = add_embeddings(matrix,labels,train_dict,'train')
    matrix,labels = add_embeddings(matrix,labels,test_dict,'test')
    matrix,labels = add_embeddings(matrix,labels,gz_dict,'G(z)')

    return np.asarray(matrix),labels

def create_plot(args):
    matrix,labels = create_matrix(args)

    # PCA on the matrix (Suggested :
    if args.dataset == 'celeba':
        pca = PCA(n_components = 50)
        matrix_pca = pca.fit_transform(X=matrix)

    image_path = os.path.join(os.getcwd(),'viz',args.dataset,'run_{}'.format(args.run),'3_vec_{}.png'.format(args.model.lower()))

    # Legend positioning
    fig, ax = plt.subplots(ncols=1)
    ax.legend(loc="upper left", bbox_to_anchor=(0.6,0.5))

    if args.dataset == 'celeba':
        hyp.plot(x=matrix_pca,
                hue=labels,
                legend = ['Closest Train Images','Test Image Embeddings','G(z) Image Embeddings'],
                fmt='.',
                ndims=2,
                reduce=args.reduce.upper(),
                title='{} plot for {}'.format(args.reduce.upper(),args.model.upper()),
                save_path=image_path,
                ax = ax)
    else:
        hyp.plot(x=matrix,
                hue=labels,
                legend = ['Closest Train Images','Test Image Embeddings','G(z) Image Embeddings'],
                fmt='.',
                ndims=2,
                reduce=None,
                title='3-vector plot for {}'.format(args.model.upper()),
                save_path=image_path,
                ax = ax)


if __name__ == '__main__':
    args = build_parser()
    create_plot(args)

