import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from scipy.stats import shapiro
from scipy.stats import bartlett
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

"""
Python module to calculate statistics for all experiments

"""

models = ['dcgan','dcgan-gp','dcgan_sim','wgan','wgan-gp','dragan','dragan_no_bn','dcgan-cons']


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
    plt.close()


def generate_box_plot(df,fname,mode=None):
    """
    Box plot

    """
    plt.figure()
    ax=df.plot.box(figsize=(12,12))
    if mode == None:
        ax.set_title('Box plot for Inpainting-Original embedding cosine distances')
    else:
        ax.set_title('Box Plot for G(z)-{} embedding cosine distances'.format(mode))
    plt.savefig(fname)
    plt.close()

def check_homegenity(col1,col2):

    """
    Check whether distances computed for 2 models
    are from the same distribution

    """
    if check_normality(col1) == True and check_normality(col2) == True:
        # Check homogenity for variances -- bartlett
        print('Performing bartlett test for equal variances')
        _,p = bartlett(col1,col2)
        if p > 0.05: # Variances equal
            print('T-test with equal variances')
            _,p = ttest_ind(col1,col2,equal_var=True)
        else:
            print('T-test with unequal variances')
            _,p = ttest_ind(col1,col2,equal_var=False)

        if p > 0.05:
            print('Distributions are homogenous')
        else:
            print('Distributions are not homogenous')


    else:
        # Check homegenity for variances -- levene
        print('Performing levene test for equal variances')
        _,p = levene(col1,col2)
        if p > 0.05:
            print('Performing Mann-Whitney U test for equality of medians')
            _,p = mannwhitneyu(col1,col2)
            if p > 0.05:
                print('Distributions are homogenous')
            else:
                print('Distributions are not homogenous')
        else:
            print('Variances for non-normal data are not equal')
            _,p = mannwhitneyu(col1,col2)
            if p > 0.05:
                print('Distributions are homogenous')
            else:
                print('Distributions are not homogenous')



def check_normality(col):
    """
    Check for normality

    """
    _,p = shapiro(col)

    if p > 0.05:
        return True

    return False




