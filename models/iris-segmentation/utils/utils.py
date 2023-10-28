import matplotlib.pyplot as plt
from datetime import datetime
import torch

def show_example_pair(ipt_image, tgt_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ipt_image, cmap='gray')
    axs[1].imshow(tgt_image, cmap='gray')
    fig.show()
    plt.waitforbuttonpress(1)

def save_simple_2d_plot(x, y, title=None, xlabel=None, ylabel=None):
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y)
    if title:
        axs.set_title(title)
    if xlabel:
        axs.set_xlabel(xlabel)
    if ylabel:
        axs.set_ylabel(ylabel)
    fig.savefig(f'{title}_{datetime.now().strftime(r"%d-%m-%Y_%H-%M-%S")}.png')
