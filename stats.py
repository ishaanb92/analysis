import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro
from scipy.stats import bartlett
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

"""
Python module to calculate statistics for all experiments

"""

models = ['dcgan-gp','dcgan','dcgan_sim','dragan_bn','wgan','wgan-gp','dragan','dcgan-cons']
colors = ['darkgreen','darkgreen','darkgreen','darkgreen','maroon','maroon','darkgreen','darkgreen']

models_xticks = ['nsgan-gp','nsgan','nsgan_sim','dragan_bn','wgan','wgan-gp','dragan','nsgan-cons']

def generate_pairs():

    pairs = []
    for model,idx in zip(models,range(len(models))):
        for i in range(idx+1,len(models)):
            pair = [model,models[i]]
            pairs.append(pair)

    return pairs

def get_mean(col):
    """
    Takes in an np array/df column and calculates mean

    """
    return col.mean()

def get_var(col):
    """
    Variance in colums

    """
    return col.var()

def plot_hist(col,fname=None):
    """
    Histogram of values in a column

    """
    plt.figure()
    col.hist(bins=100)
    plt.xlabel('G(z) - Test Image Cosine Distance')
    plt.ylabel('Count')
    plt.xlim(0,2)
    plt.savefig(fname)
    plt.close('all')


def generate_box_plot(df,fname,mode=None,kwds=None):
    """
    Box plot

    """
    plt.figure()
    if kwds is None:
        ax,_=df.plot.box(figsize=(12,12),return_type='both')
    else:
        ax,barplot=df.plot.box(figsize=(12,12),return_type='both',**kwds)
        # Color GAN box plot based on divergence used
        for patch,color in zip(barplot['boxes'],colors):
            patch.set_facecolor(color)

    if mode == None:
        ax.set_title('Box plot for Inpainting-Original embedding cosine distances')
    else:
        if mode == 'gap':
            ax.set_title('Box Plot for gap between test and train distances')
        else:
            ax.set_title('Box Plot for G(z)-{} embedding cosine distances'.format(mode))

    ax.set_xlabel('GAN Models')

    ax.set_ylabel('Cosine Distance')

    plt.savefig(fname)
    plt.close('all')

def check_homegenity(col1,col2,verbose=False):

    """
    Check whether distances computed for 2 models
    are from the same distribution

    """
    if check_normality(col1) == True and check_normality(col2) == True:
        # Check homogenity for variances -- bartlett
        if verbose is True:
            print('Performing bartlett test for equal variances')
        _,p = bartlett(col1,col2)
        if p > 0.05: # Variances equal
            if verbose is True:
                print('T-test with equal variances')
            _,p = ttest_ind(col1,col2,equal_var=True)
        else:
            if verbose is True:
                print('T-test with unequal variances')
            _,p = ttest_ind(col1,col2,equal_var=False)

        if p > 0.05:
            if verbose is True:
                print('Distributions are homogenous')
            return True
        else:
            if verbose is True:
                print('Distributions are not homogenous')
            return False


    else:
        # Check homegenity for variances -- levene
        if verbose is True:
            print('Performing levene test for equal variances')
        _,p = levene(col1,col2)
        if p > 0.05:
            if verbose is True:
                print('Performing Mann-Whitney U test for equality of medians')
            _,p = mannwhitneyu(col1,col2)
            if p > 0.05:
                if verbose is True:
                    print('Distributions are homogenous')
                return True
            else:
                if verbose is True:
                    print('Distributions are not homogenous')
                return False
        else:
            if verbose is True:
                print('Variances for non-normal data are not equal')
            _,p = mannwhitneyu(col1,col2)
            if p > 0.05:
                if verbose is True:
                    print('Distributions are homogenous')
                return True
            else:
                if verbose is True:
                    print('Distributions are not homogenous')
                return False




def check_normality(col):
    """
    Check for normality

    """
    _,p = shapiro(col)

    if p > 0.05:
        return True

    return False




