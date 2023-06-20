import timeit
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from Source.ConstSectBeamCol.File.IO import load_model
from Source.ConstSectBeamCol.Variables import Model
from Source.ConstSectBeamCol.Training.Train import train_model
import matplotlib


def Run():
    matplotlib.use('TkAgg')
    res_folder = Model.OutResult.Folder + Model.OutResult.ModelName + ".rst"
    path = os.getcwd()
    model = load_model(path + "\\Examples\\pre-train.rst", 'pre-train')
    model.train()

    opt = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.999))
    loss = nn.MSELoss()

    num_epochs = Model.Analysis.num_epochs
    num_sample = Model.Analysis.num_sample

    # Training loop
    desc = "start training..."
    pbar = tqdm(range(num_epochs), desc=desc)

    load_factor = 1
    tol = Model.Analysis.TOL
    train_model(model, res_folder, Model.OutResult.ModelName, load_factor, num_sample, tol, loss, opt, pbar)
