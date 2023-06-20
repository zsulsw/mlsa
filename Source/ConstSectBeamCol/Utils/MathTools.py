import torch


def gradients(U, X, order=1):
    if order == 1:
        return torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True,
                                   allow_unused=True)[0]
    else:
        return gradients(gradients(U, X), X, order=order - 1)


def Integration1D(U, X, LB, UB):
    TempX = X.squeeze()
    TempU = U.squeeze()
    TempX, Index = torch.sort(TempX)
    TempU = TempU[Index]
    Index1 = TempX >= LB
    Index2 = TempX <= UB
    Index = Index1 * Index2
    TempX = TempX[Index]
    TempU = TempU[Index]
    Integral = torch.trapz(TempU, TempX)
    return Integral
