import os,pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from stats import *
import matplotlib.pyplot as plt


def read_dict(model):
    with open(os.path.join(os.getcwd(),'viz','embeddings','{}_emb_dict.pkl'.format(model.lower())),'rb') as f:
        return pickle.load(f)




def sanity_checks(model):
    emb_dict = read_dict(model)
    embs = np.asarray(list(emb_dict.values()))
    #pca = PCA(n_components = 'mle',svd_solver='full')
    #pca.fit(embs)
    #emb_red = pca.transform(embs)
    #print('{} :: Original Shape : {} PCA Reduction : {}'.format(model.upper(),embs.shape,emb_red.shape))

    # Create box-plot of pair-wise distances
    return create_pairwise_box_plot(X=embs,model=model)

def create_pairwise_box_plot(X,model):
    """
    Given an array of vectors X, calculate pair-wise distances

    """
    remove_duplicates = []
    distance_matrix = pairwise_distances(X,metric='cosine')

    for i in range(distance_matrix.shape[0]):
        for j in range(i):
            remove_duplicates.append(distance_matrix[i][j])

    return np.asarray(remove_duplicates)

def cluster(model):
    emb_dict = read_dict(model)
    embs = np.array(list(emb_dict.values()))

    sil_scores = []
    for n_clus in range(2,11): # Increase cluster size from 1-10
        kmeans = KMeans(n_clusters = n_clus)
        kmeans.fit(embs)
        # Compute sil scores
        sil_scores.append(silhouette_score(X=embs,labels=kmeans.labels_))

    # Silhoutte Scores : Near 1 --> Well separated clusters
    #                    Near 0 --> Overlapping clusters

    x_axis = [i for i in range(2,11)]
    plt.figure()
    plt.plot(x_axis,sil_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhoutte Score')
    plt.savefig('viz/{}_cluster_scores.png'.format(model.lower()))
    plt.close()


def do_sanity():
    pair_wise = []
    for model in models:
        distance_array = sanity_checks(model)
        pair_wise.append(distance_array)

    pair_wise = np.array(pair_wise)
    pair_wise_t = np.transpose(pair_wise)

    df = pd.DataFrame(data=pair_wise_t,columns=models)
    plt.figure()
    df.plot.box(figsize=(10,10))
    plt.savefig('viz/pairwise_box_gz.png')
    plt.close()

if __name__ == '__main__':
    for model in models:
        cluster(model)



