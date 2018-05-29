import os,pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
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

    # Create histogram of pair-wise distances
    return create_pairwise_hist(X=embs,model=model)

def create_pairwise_hist(X,model):
    """
    Given an array of vectors X, calculate pair-wise distances

    """
    remove_duplicates = []
    distance_matrix = pairwise_distances(X,metric='cosine')

    for i in range(distance_matrix.shape[0]):
        for j in range(i):
            remove_duplicates.append(distance_matrix[i][j])


    plt.figure()
    plt.hist(x=np.asarray(remove_duplicates),bins=100)
    plt.savefig('{}_distance_hist.png'.format(model.lower()))
    plt.close()

    return np.asarray(remove_duplicates)

if __name__ == '__main__':

    pair_wise = []
    for model in models:
        distance_array = sanity_checks(model)
        pair_wise.append(distance_array)

    pair_wise = np.array(pair_wise)
    print(pair_wise.shape)
    pair_wise_t = np.transpose(pair_wise)
    print(pair_wise_t.shape)



    df = pd.DataFrame(data=pair_wise_t,columns=models)
    plt.figure()
    df.plot.box(figsize=(10,10))
    plt.savefig('pairwise_box.png')
    plt.close()


