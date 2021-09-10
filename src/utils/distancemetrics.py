import numpy as np
from src.evolution.representation import Representation

def euclidean(repa:Representation, repb:Representation):
    return np.linalg.norm(repa.get_output(10,10)-repb.get_output(10,10))

def nuclearnorm(repa:Representation, repb:Representation):
    outa = repa.get_output(10,10)
    outb = repb.get_output(10,10)
    score = 0
    for i in range(outa.shape[-1]):
        score += np.linalg.norm(outa[:,:,i] - outb[:,:,i],ord='nuc')
    return score


def variance(repa:Representation, repb:Representation):
    dim = 100
    return np.var(repa.get_output(dim,dim)-repb.get_output(dim,dim))
