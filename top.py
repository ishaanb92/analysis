#!/usr/bin/env python3

from analyze_embeddings import *
from analyze_metric import *
from stats import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser
"""
Top-level scripts that produces results for all runs and writes them to file
Calculates metric/embedding statistics across multiple runs

"""

def calculate_mean_stats(last_run,draw,log_file):
    """
    Calculates mean stats over all runs

    """
    sim_metric_means = [] # List of dicts from each run
    emb_test_dist_means = []
    emb_train_dist_means = []

    for run in range(2,last_run+1):
        log_file.write('### Analysis for Run {} ### \n\n'.format(run))
        log_file.write('*** Embeddings Analysis ***\n')
        test_means_dict,train_means_dict,_,_ = analyze_embeddings(run=str(run),draw=draw,log_file=log_file)
        emb_test_dist_means.append(test_means_dict)
        emb_train_dist_means.append(train_means_dict)
        log_file.write('\n')
        log_file.write('*** Metric Analysis ***\n')
        run_mean_dict = analyze_metric(run=str(run),draw=draw,log_file=log_file)
        sim_metric_means.append(run_mean_dict)
        log_file.write('\n\n\n')

    sim_metric_mean = {}
    avg_gap = {}


    # Statistics across runs
    for model in models:
        sim_means = []
        emb_test_means = []
        emb_train_means = []
        log_file.write('Inter-run analysis for {}\n'.format(model.upper()))

        for sim_means_dict,emb_test_dict,emb_train_dict in zip(sim_metric_means,emb_test_dist_means,emb_train_dist_means):
            sim_means.append(sim_means_dict[model])
            emb_test_means.append(emb_test_dict[model])
            emb_train_means.append(emb_train_dict[model])

        # Analyze
        sim_metric_mean[model] = np.asarray(sim_means).mean()
        emb_test_mean = np.asarray(emb_test_means).mean()
        emb_train_mean = np.asarray(emb_train_means).mean()

        sim_metric_std = np.asarray(sim_means).std()
        emb_test_std = np.asarray(emb_test_means).std()
        emb_train_std = np.asarray(emb_train_means).std()

        avg_gap[model] = math.fabs(emb_test_mean-emb_train_mean)

        log_file.write('Similarity Metric  Original - Inpainting ::  Mean = {} Std = {}\n'.format(np.around(sim_metric_mean[model],decimals=4),np.around(sim_metric_std,decimals=4)))
        log_file.write('Embeddings Metric Test - G(z) :: Mean = {} Std = {}\n'.format(np.around(emb_test_mean,decimals=4),np.around(emb_test_std,decimals=4)))
        log_file.write('Embeddings Metric Train - G(z) :: Mean = {} Std = {}\n'.format(np.around(emb_train_mean,decimals=4),np.around(emb_train_std,decimals=4)))
        log_file.write('Overfitting gap = {}\n'.format(np.around(avg_gap[model],decimals=4)))
        log_file.write('\n\n')

    #Plot a joint bar graph
    sim_means = []
    gaps = []
    for model in models:
        sim_means.append(sim_metric_mean[model])
        gaps.append(avg_gap[model])

    x = [i for i in range(len(models))]
    w = 0.3
    x = np.asarray(x)

    plt.figure(figsize=(15,15))
    bar1 = plt.bar(x, sim_means,width=w,color='b',align='center')
    bar2 = plt.bar(x+w,gaps,width=w,color='g',align='center')
    models_u = [model.upper() for model in models]
    plt.xticks(x,models_u)
    plt.ylabel('Cosine Distances')
    plt.legend([bar1,bar2],['Mean Cosine Distance','Mean Generalization Gap'])
    plt.xlabel('GAN Models')
    plt.savefig('joint_bar.png')
    plt.close('all')


def calculate_ci(col):
    """
    Given a column of similarity scores, calculates
    the 95% CI

    """

    std = np.sqrt(get_var(col))
    ci = np.multiply(1.96,std)
    return ci

def accumulate_scores(last_run,draw,log_file):

    """
    Accumulates scores/distances from different runs
    and generates box-plot

    """
    # Similarity -- concat all data-frames
    df_list = []
    for run in range(2,last_run+1):
        df = pd.read_csv(os.path.join(os.getcwd(),'celebA_metric_results','run_{}'.format(run),'gan_distances.csv'))
        df_list.append(df)

    df_concat = pd.concat(df_list,axis=0) # Merge all data frames vertically

    valid_cols = []

    for model in models:
        valid_cols.append(df_concat[model.upper()])

    df_valid_cols = pd.concat(valid_cols,axis=1) # Merge valid columns horizontally

    if draw is True:
        kwds = {}
        kwds['patch_artist'] = True
        generate_box_plot(df=df_valid_cols,fname='sim_distances.png',kwds=kwds)

    log_file.write('****CI for Image Similarity scores****\n')
    # CI
    for model in models:
        ci = calculate_ci(df_valid_cols[model.upper()])
        mean = get_mean(df_valid_cols[model.upper()])
        std = np.sqrt(get_var(df_valid_cols[model.upper()]))
        log_file.write('Model:{} Mean : {} Std : {}\n'.format(model.upper(),np.around(mean,decimals=4),np.around(std,decimals=4)))
        if check_normality(df_valid_cols[model.upper()]) is True:
            upper_ci = mean + ci
            lower_ci = mean - ci
            log_file.write('Model : {} Mean : {} Std: {}\n'.format(model.upper(),mean,lower_ci,upper_ci))
        else:
            log_file.write('Scores for {} form a non-normal distribution\n'.format(model.upper()))




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--last_run',type=int,help='Value of the last run for which results are available',required=True)
    parser.add_argument('--draw',action='store_true',help='Flag to regenerate figures for runs',default=False)

    args = parser.parse_args()

    last_run = args.last_run
    draw = args.draw

    log_file = open('log_file.txt','w')
    log_file.write('****Analysis for In-painting experiments*****\n')
    log_file.write('This file has been automatically generated, to regenerate or add data from more runs, re-run top.py\n')
    log_file.write('\n\n\n\n')

    calculate_mean_stats(last_run=last_run,draw=draw,log_file=log_file)
    accumulate_scores(last_run=last_run,draw=draw,log_file=log_file)

    log_file.close()
