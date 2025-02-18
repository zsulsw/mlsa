###########################################################################################
# Machine Learning-based Second-order Analysis of Beam-columns through PINNs
# Developed by Siwei Liu, Liang Chen and Haoyi Zhang
# License: GPL-3.0
###########################################################################################

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from Source.File.IO import load_model, init_load_deform_res
from Source.Variables import Model
from Source.Training.Train import train_model
import matplotlib


def Run(device):
    matplotlib.use('TkAgg')
    res_folder = Model.OutResult.Folder + Model.OutResult.ModelName + ".rst"
    path = os.getcwd()
    model = load_model(path + "\\NeuralNetwork\\" + Model.Analysis.SelectNN, Model.Analysis.SelectNN, None, device)
    model.train()

    opt = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.999))
    loss = nn.MSELoss()

    num_epochs = Model.Analysis.num_epochs
    num_sample = Model.Analysis.num_sample

    # Training loop
    desc = "start training..."
    pbar = tqdm(range(num_epochs), desc=desc)

    # Total load steps
    step_num = Model.Analysis.load_step
    load_factor_inc = Model.Analysis.target_LF / step_num
    load_factor = load_factor_inc

    # Start training for each load step
    init_load_deform_res(res_folder, Model.OutResult.ModelName)
    while load_factor <= 1:
        tol = Model.Analysis.TOL * load_factor
        train_model(model, res_folder, Model.OutResult.ModelName, load_factor, num_sample, tol, loss, opt, pbar, device)
        load_factor += load_factor_inc
