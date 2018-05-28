import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

"""
Python module to calculate statistics for all experiments

"""

models = ['dcgan','dcgan-gp','wgan','wgan-gp','dragan','dcgan-cons']


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

def plot_hist(col,model=None,fname=None,indx=0):
    """
    Histogram of values in a column

    """
    plt.figure(indx)
    col.hist(bins=100)
    plt.xlabel('G(z) - Test Image Cosine Distance')
    plt.ylabel('Count')
    plt.xlim(0,2)
    plt.savefig('viz/{}_{}.png'.format(model.upper(),fname))



