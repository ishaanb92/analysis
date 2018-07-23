import numpy as np
import pickle
import matplotlib.pyplot as plt
import shutil
import os
from argparse import ArgumentParser
from sys import exit
from stats import models

"""
Analysis script to check the 'effect' of regularization
DRAGAN/WGAN-GP reg terms seek to constrain the disc/critic gradient to have unit norm

How effective is this term in accomplishing this goal?

"""

def flatten_grads(grads):
    grad_list = []
    for iteration in grads: # Each iteration
            grad_list.append(iteration.mean()) # For a given iteration, append avg value of gradient norm over all batches

    return grad_list


def plot_grads(grads,model,gp_term = False):

    """
    Plots norm of gradient v/s iteration of training
    gp_term : True => Norm of GP reg term for unregularized DCGAN
              False => Norm of DRAGAN reg term for uregularized DCGAN
    """

    iters = [i*10 for i in range(len(grads))] # X-axis containing iteration number


    grad_list = flatten_grads(grads)

    plt.figure()

    plt.plot(iters,grad_list)

    plt.xlabel('Training Iteration')

    plt.title('{} : Gradient Norm used in Regularization term'.format(model.upper()))


    if model == 'dcgan' or model == 'wgan' or model == 'dcgan_sim':
        pass
    else:
        plt.ylim((0,5))


    if model == 'wgan':
        plt.ylabel('Unregularized Critic Gradient Norm')
    elif model == 'wgan-gp':
        plt.ylabel('Critic Gradient Norm')
    elif model == 'dcgan' or model == 'dcgan_sim':
        plt.ylabel('Unregularized Discriminator Gradient Norm')
    else:
        plt.ylabel('Discriminator Gradient Norm')

    if os.path.exists('grad_figures') is False:
        os.makedirs('grad_figures')

    if gp_term is False:
        filename = 'grads_{}'.format(str(model)) +'.png'
    else:
        filename = 'grads_{}_gp'.format(str(model)) +'.png'

    save_path = os.path.join('grad_figures',filename)
    plt.savefig(save_path)

    plt.close('all')

def create_plots(model):

    pkl_path = os.path.join('grad_norms','{}_grads.pkl'.format(model.lower()))
    with open(pkl_path,'rb') as f:
        grads = pickle.load(f)
    plot_grads(grads,model)
    # Plot the GP term for un-reg DCGAN
    if model == 'dcgan':
        pkl_path = os.path.join('grad_norms','{}_grads_gp.pkl'.format(model.lower()))
        with open(pkl_path,'rb') as f:
            grads = pickle.load(f)
        plot_grads(grads,model,gp_term=True)



if __name__ == '__main__':
    for model in models:
        if model == 'dcgan-cons':
            continue
        create_plots(model)








