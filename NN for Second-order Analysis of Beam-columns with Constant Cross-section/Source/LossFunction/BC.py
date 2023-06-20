import torch
import numpy as np
from Source.Utils.MathTools import gradients


def force_eq(s, Delta, Theta, loss, E, A, I, L, P1, Fy1, Vyq, LF, Impf):
    P = LF * P1 * torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) - Fy1 * torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) + Vyq * torch.sin(Theta)
    V = -LF * P1 * torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) - Fy1 * torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) + Vyq * torch.cos(Theta)
    # =================================================
    return(loss(gradients(Delta, s, 1), torch.ones_like(s) * P * L / (E * A)),
           loss(gradients(Theta, s, 2), torch.ones_like(s) * V * L ** 2 / (E * I)),
           P, V)


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
