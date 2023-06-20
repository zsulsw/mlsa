import torch
import numpy as np
from Source.Utils.MathTools import gradients


def ux_i(Delta, loss, bc, device):
    return bc * loss(Delta[0], torch.zeros(1, 1, device=device))


def ux_j(x1, loss, bc, device):
    return bc * loss(x1, torch.zeros(1, 1, device=device))

def uy_j(y1, loss, bc, device):
    return bc * loss(y1, torch.zeros(1, 1, device=device))


def rz_i(s, Theta, loss, E, I, L, M0, LF, bc, device):
    return bc * loss(Theta[0], torch.zeros(1, 1, device=device)) + (1 - bc) * loss(gradients(Theta, s, 1)[0], torch.ones(1, 1, device=device) * LF * M0 * L / (E * I))


def rz_j(s, Theta, loss, E, I, L, M1, LF, bc, device):
    return bc * loss(Theta[-1], torch.zeros(1, 1, device=device)) + (1 - bc) * loss(gradients(Theta, s, 1)[-1], torch.ones(1, 1, device=device) * LF * M1 * L / (E * I))