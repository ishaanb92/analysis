import os,pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from stats import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def read_gz_dict(model,run):

    with open(os.path.join(os.getcwd(),'viz','run_{}'.format(run),'embeddings','{}_emb_dict.pkl'.format(model.lower())),'rb') as f:
        gz_dict = pickle.load(f)
        return np.array(list(gz_dict.values()))


def read_train_dict(model,run):

    with open(os.path.join(os.getcwd(),'viz','run_{}'.format(run),'{}_embedding'.format(model.upper()),'closest_train_emb.pkl'),'rb') as f:
        train_emb_dict =  pickle.load(f)
        return np.array(list(train_emb_dict.values()))


def compute_pairwise_distances(X,model):
    """
    Given an array of vectors X, calculate pair-wise distances
    Array contains 128-D embeddings of G(z) produced by the model

    """
    remove_duplicates = []
    distance_matrix = pairwise_distances(X,metric='euclidean')

    for i in range(distance_matrix.shape[0]):
        for j in range(i):
            remove_duplicates.append(distance_matrix[i][j])

    return np.asarray(remove_duplicates)

def pairwise_analysis(root_dir,run):
    """
    Creates a box plot of pair-wise distances between the
    closest training image enbeddings found for all models

    """
    pair_wise = []
    for model in models:

        distance_array = compute_pairwise_distances(read_gz_dict(model,run),model)
        pair_wise.append(distance_array)

    pair_wise = np.array(pair_wise)
    pair_wise_t = np.transpose(pair_wise)

    df = pd.DataFrame(data=pair_wise_t,columns=models)
    plt.figure()
    df.plot.box(figsize=(10,10))
    plt.savefig(os.path.join(root_dir,'pairwise_box_gz.png'))
    plt.close()


def cluster(root_dir,model,run,dim_red=True):
    """
    Measures number of clusters using the silhouette score

    """
    embs = read_gz_dict(model=model,run=run)

    if dim_red is True:
        pca = PCA(n_components=5)
        embs = pca.fit_transform(embs)
        root_dir = root_dir + '_reduce'
        if os.path.exists(root_dir) is False:
            os.makedirs(root_dir)

    sil_scores = []
    for n_clus in range(2,11): # Increase cluster size from 2-10
        kmeans = KMeans(n_clusters = n_clus)
        kmeans.fit(embs)
        # Compute sil scores
        sil_scores.append(silhouette_score(X=embs,labels=kmeans.labels_))

    # Silhoutte Scores : Near 1 --> Well separated clusters
    #                    Near 0 --> Overlapping clusters

    x_axis = [i for i in range(2,11)]
    plt.figure()
    plt.plot(x_axis,sil_scores)
    plt.ylim(0,1)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhoutte Score')
    plt.savefig(os.path.join(root_dir,'{}_cluster_scores.png'.format(model.lower())))
    plt.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run',type=str,help='Experiment #')
    args = parser.parse_args()

    pairwise_root = os.path.join(os.getcwd(),'viz','run_{}'.format(args.run),'pairwise')
    cluster_root = os.path.join(os.getcwd(),'viz','run_{}'.format(args.run),'cluster')

    if os.path.exists(pairwise_root) is False:
        os.makedirs(pairwise_root)

    if os.path.exists(cluster_root) is False:
        os.makedirs(cluster_root)

    pairwise_analysis(pairwise_root,args.run)

    for model in models:
        cluster(cluster_root,model,args.run,dim_red=True)
        cluster(cluster_root,model,args.run,dim_red=False)



