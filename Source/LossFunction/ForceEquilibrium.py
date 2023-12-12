###########################################################################################
# Machine Learning-based Second-order Analysis of Beam-columns through PINNs
# Developed by Siwei Liu, Liang Chen and Haoyi Zhang
# License: GPL-3.0
###########################################################################################

import torch
import numpy as np
from Source.Utils.MathTools import gradients


def force_eq(s, Delta, Theta, loss, E, A, I, L, P1, Fy1, Vyq, LF, Impf):
    P = LF * P1 * torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) - Fy1 * torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) + Vyq * torch.sin(Theta)
    V = -LF * P1 * torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) - Fy1 * torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) + Vyq * torch.cos(Theta)
    # =================================================
    return(loss(gradients(Delta, s, 1), torch.ones_like(s) * P * L / (E * A)),
           loss(gradients(Theta, s, 2), torch.ones_like(s) * V * L ** 2 / (E * I)), P, V)
