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


def generate_box_plot(df,fname):
    """
    Box plot

    """
    plt.figure()
    ax=df.plot.box()
    ax.set_title('Box Plot for G(z)-Test embedding cosine distances')
    plt.savefig(fname)




