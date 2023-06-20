import torch


def GetLOSS(LOSS_List, LossWeight):
    LOSS = LossWeight[0] * LOSS_List[0]
    LOSS_value = LOSS_List[0].item()
    for ii in range(len(LOSS_List) - 1):
        LOSS = LOSS + LossWeight[ii + 1] * LOSS_List[ii + 1]
        LOSS_value = LOSS_value + LOSS_List[ii + 1].item()
    return LOSS, LOSS_value


def UpdateLossWeight(model, LOSS_List, LossWeight, beta=0.9):
    Last_LossWeight = torch.tensor(LossWeight).clone().detach()
    L1 = LOSS_List[0]
    model.zero_grad()
    tgrad = ObtainGrad(model, L1)
    max_value = torch.max(torch.abs(tgrad))
    mean_value = []
    for ii in range(len(LossWeight) - 1):
        model.zero_grad()
        tL = LOSS_List[ii + 1]
        tgrad = ObtainGrad(model, tL)
        tmean_value = torch.mean(torch.abs(tgrad))
        mean_value.append(tmean_value)
        tLossWeight = 1 / LossWeight[ii + 1] * max_value / tmean_value
        LossWeight[ii + 1] = beta * LossWeight[ii + 1] + (1-beta) * tLossWeight
    if torch.isinf(torch.tensor(LossWeight)).any() or torch.isnan(torch.tensor(LossWeight)).any():
        return Last_LossWeight
    return torch.clamp(torch.tensor(LossWeight), max=100, min=1)


def ObtainGrad(model, L, retain_graph=True):
    L.backward(retain_graph=retain_graph)
    ii = 0
    for name, parms in model.named_parameters():
        tgrad = parms.grad
        if len(tgrad.size()) == 1:
            size = tgrad.size()[0]
        else:
            size = tgrad.size()[0] * tgrad.size()[1]
        if ii == 0:
            grad_value = tgrad.view(size,1)
            ii += 1
            continue
        tgrad = tgrad.view(size,1)
        grad_value = torch.cat((grad_value, tgrad))
        # grad_value = grad_value.view(len(grad_value), 1)
    return grad_value
