import torch
import numpy as np
from matplotlib import pyplot as plt
from Source.Utils.MathTools import gradients, Integration1D


def start_train():
    plt.ion()
    plt.rc('font', size=8)  # controls default text sizes
    plt.rc('axes', titlesize=8)  # fontsize of the axes title
    plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=8)  # legend fontsize
    plt.rc('figure', titlesize=8)  # fontsize of the figure title
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(8, 10))
    return fig


def process(fig, s, Delta, Theta, E, A, I, L, P, V, P1, Fy1, M1, Vyq, LF, Impf):
    plt.figure(fig.number)
    plt.clf()
    DDeltaDs = gradients(Delta, s, 1)
    DThetaDs = gradients(Theta, s, 1)
    D2ThetaDs2 = gradients(Theta, s, 2)
    x = torch.zeros_like(s)
    y = torch.zeros_like(s)
    Mq = torch.zeros_like(s)
    for i in range(len(s)):
        x[i] = Integration1D(L * torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) - torch.cos(Theta + np.pi * Impf * torch.cos(np.pi * s)) * gradients(Delta, s, 1), s, 0, s[i])
        y[i] = Integration1D(L * torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) - torch.sin(Theta + np.pi * Impf * torch.cos(np.pi * s)) * gradients(Delta, s, 1), s, 0, s[i])
    for i in range(len(s)):
        Mq[i] = Integration1D(-Vyq, x, 0, x[-1]) - Integration1D(-Vyq, x, 0, x[i])
    M = Mq - (LF * P1 * y - Fy1 * (x[-1] - x)) + LF * M1
    x = x.cpu()
    y = y.cpu()
    Delta = Delta.cpu()
    Theta = Theta.cpu()
    DDeltaDs = DDeltaDs.cpu()
    DThetaDs = DThetaDs.cpu()
    D2ThetaDs2 = D2ThetaDs2.cpu()
    s = s.cpu()
    P = P.cpu()
    V = V.cpu()
    M = M.cpu()
    x = x.detach().numpy()
    y = y.detach().numpy()
    Delta = Delta.detach().numpy()
    Theta = Theta.detach().numpy()
    DDeltaDs = DDeltaDs.detach().numpy()
    DThetaDs = DThetaDs.detach().numpy()
    D2ThetaDs2 = D2ThetaDs2.detach().numpy()
    s = s.detach().numpy()
    P = P.detach().numpy()
    V = V.detach().numpy()
    M = M.detach().numpy()
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(s * L, Delta)
    ax1.legend(["Delta"])
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(s * L, Theta)
    ax2.legend(["Theta"])
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(s * L, DDeltaDs)
    ax3.plot(s * L, np.ones_like(s) * P * L / (E * A))
    ax3.legend(["DDeltaDs", "P * L / (E * A)"])
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(s * L, D2ThetaDs2)
    ax4.plot(s * L, np.ones_like(s) * V * L ** 2 / (E * I))
    ax4.legend(["D2ThetaDs2", "V * L ** 2 / (E * I)"])
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(s * L, DThetaDs)
    ax5.plot(s * L, M * L / (E * I))
    ax5.legend(["DThetaDs", "M * L / (E * I)"])
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(x, y)
    ax6.legend(["Deformation"])
    plt.pause(0.1)


def end_train(fig):
    plt.figure(fig.number)
    plt.ioff()
    plt.close()
