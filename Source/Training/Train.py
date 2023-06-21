import torch
import numpy as np
from Source.LossFunction.ForceEquilibrium import force_eq
from Source.LossFunction.BoundaryConditions import ux_i, ux_j, uy_j, rz_i, rz_j
from Source.Utils import AdaoptiveLossWeight
from Source.File.IO import save_model
from Source.Visualization.BeamColumn import start_train, process, end_train
from Source.Variables import Model
from Source.Utils.MathTools import Integration1D, gradients


def train_model(model, model_path, model_name, load_factor, num_sample=100, TOL=1E-12, loss=torch.nn.MSELoss, opt=None,
                pbar=None, device='cpu'):
    model = model.to(device)
    MinLoss = 1
    LossWeight = [1, 1, 1, 1, 1, 1, 1]
    fig = start_train()
    for epoch in pbar:
        # Place sampling points, S, along the axial length
        s = torch.rand(int(num_sample) - 2, 1).to(device)
        s, s_indice = torch.sort(s, dim=0)
        s = torch.cat((s, torch.tensor([[1]], device=device)))
        s = torch.cat((torch.tensor([[0]], device=device), s))
        s.requires_grad = True
        res = model(s)
        Delta, Theta = res[:, 0], res[:, 1]
        Delta = Delta.unsqueeze(-1)
        Theta = Theta.unsqueeze(-1)
        member_id = next(_ for _ in Model.Member.ID)
        E = Model.Material.E[Model.Section.MatID[Model.Member.SectID[member_id]]]
        A = Model.Section.A[Model.Member.SectID[member_id]]
        I = Model.Section.Iy[Model.Member.SectID[member_id]]
        L = Model.Member.L0[member_id]
        P1 = Model.JointLoad.FX[Model.Member.J[member_id]]
        q1 = Model.MemberUDL.QY1[member_id]
        q2 = Model.MemberUDL.QY2[member_id]
        try:
            M0 = Model.JointLoad.MZ[Model.Member.I[member_id]]
        except:
            M0 = 0
        M1 = Model.JointLoad.MZ[Model.Member.J[member_id]]
        Impf = Model.Member.Imperfection[member_id]
        Vyq = ((1 - s) * load_factor * q1 + (1 + s) * load_factor * q2) / 2 * (1 - s) * L
        Fy1 = (-(gradients(Theta, s, 2)[-1]) * E * I / L ** 2 - load_factor * P1 * torch.sin(Theta[-1] + np.pi * Impf * torch.cos(np.pi * s[-1]))) / torch.cos(Theta[-1] + np.pi * Impf * torch.cos(np.pi * s[-1])) * Model.Boundary.UY[2]
        x1 = Integration1D(L * torch.cos(Theta) - torch.cos(Theta) * gradients(Delta, s, 1), s, 0, s[-1])
        y1 = Integration1D(L * torch.sin(Theta) - torch.sin(Theta) * gradients(Delta, s, 1), s, 0, s[-1])
        L1, L2, P, V = force_eq(s, Delta, Theta, loss, E, A, I, L, P1, Fy1, Vyq, load_factor, Impf)
        L3 = ux_i(Delta, loss, Model.Boundary.UX[1], device)
        L4 = ux_j(x1, loss, Model.Boundary.UX[2], device)
        L5 = uy_j(y1, loss, Model.Boundary.UY[2], device)
        L6 = rz_i(s, Theta, loss, E, I, L, M0, load_factor, Model.Boundary.RZ[1], device)
        L7 = rz_j(s, Theta, loss, E, I, L, M1, load_factor, Model.Boundary.RZ[2], device)
        # =================================================
        # Optimize the model parameters
        opt.zero_grad()
        LossFuncSum = [L1, L2, L3, L4, L5, L6, L7]
        LOSS, LOSS_value = AdaoptiveLossWeight.GetLOSS(LossFuncSum, LossWeight)
        if LOSS_value < TOL:
            process(fig, s, Delta, Theta, E, A, I, L, P, V, P1, Fy1, M1, Vyq, load_factor, Impf)
        if epoch % 1000 == 0 or (epoch % 500 == 0 and epoch <= 1000):
            process(fig, s, Delta, Theta, E, A, I, L, P, V, P1, Fy1, M1, Vyq, load_factor, Impf)
        if epoch % 20 == 0 and epoch > 10:
            print("LossWeight:", LossWeight)
            print([L1.item(), L2.item(), L3.item(), L4.item(), L5.item(), L6.item(), L7.item()])
            opt.zero_grad()
            LOSS.backward()
            opt.step()
        else:
            LOSS.backward()
            opt.step()
        if epoch % 100 == 0 and epoch != 0 and LOSS_value <= MinLoss:
            save_model(model, model_path, model_name)
        if LOSS_value < TOL:
            break
        pbar.set_description("Loss %s" % LOSS.item())
    end_train(fig)
    print("Current loss value:", LOSS.item())
    save_model(model, model_path, model_name)
    model.eval()
