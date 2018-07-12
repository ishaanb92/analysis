import numpy as np
import pickle
import matplotlib.pyplot as plt
import shutil
import os
from argparse import ArgumentParser
from sys import exit

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


def plot_grads(grads,model):

    """
    Plots norm of gradient v/s iteration of training
    """

    iters = [i*10 for i in range(len(grads))] # X-axis containing iteration number


    grad_list = flatten_grads(grads)

    plt.figure()

    plt.plot(iters,grad_list)

    plt.xlabel('Training Iteration')

    plt.title('{} : Gradient Norm used in Regularization term'.format(model.upper()))

    if model == 'wgan-gp':
        plt.ylabel('Critic Gradient Norm')
    else:
        plt.ylabel('Discriminator Gradient Norm')

    if os.path.exists('grad_figures') is False:
        os.makedirs('grad_figures')

    filename = 'grads_{}'.format(str(model)) +'.png'
    save_path = os.path.join('grad_figures',filename)
    plt.savefig(save_path)

    plt.close('all')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the GAN model', required=True)

    args = parser.parse_args()

    pkl_path = '{}_grads.pkl'.format(args.model.lower())

    with open(pkl_path,'rb') as f:
        grads = pickle.load(f)

    plot_grads(grads=grads,model=args.model.lower())








