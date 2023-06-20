from test.LSSI.PINN_Model import BaseNetwork
import timeit
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from Source.File.IO import load_model
from Source.Variables import Model
from Source.Training.Train import train_model
from Source.File import ReadData
import matplotlib

def Run():
    matplotlib.use('TkAgg')
    res_folder = Model.OutResult.Folder + Model.OutResult.ModelName + ".rst"
    conf_file = res_folder +"\\"+ Model.OutResult.ModelName + ".config"
    tar_file = res_folder +"\\" + Model.OutResult.ModelName + ".tar"
    FileName = res_folder + "\\" + Model.OutResult.ModelName + ".txt"
    #with open(FileName, 'w') as f:
        #for i in range(1):
            #print("Loop: {}".format(i))
    #for i in range(1, int(Model.Analysis.load_step) + 1):

    #if os.path.exists(conf_file) and os.path.exists(tar_file):
        #model = load_model(res_folder, Model.OutResult.ModelName)
    #else:
    path = os.getcwd()
    model = load_model(path + "\\Examples\\pre-train.rst", 'pre-train')
    #model = BaseNetwork(act_fn=[torch.nn.Tanhshrink(), torch.nn.Tanh(), torch.nn.Tanhshrink()],
                            #input_size=1, output_size=2, hidden_sizes=[200, 200, 200])

    model.train()

    opt = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.999))
    loss = nn.MSELoss()

    num_epochs = Model.Analysis.num_epochs
    num_sample = Model.Analysis.num_sample
    target_LF = Model.Analysis.target_LF

    # Training loop
    desc = "start training..."
    pbar = tqdm(range(num_epochs), desc=desc)

    StartTime = timeit.default_timer()
    #f.write("Loop: {}\n".format(i))
    #load_factor = target_LF / Model.Analysis.load_step * i
    load_factor = 1
    print("\nLoad Factor: {}\n".format(load_factor))
    #tol = Model.Analysis.TOL / Model.Analysis.load_step * i
    tol = Model.Analysis.TOL
    train_model(model, res_folder, Model.OutResult.ModelName, load_factor, num_sample, tol, loss, opt, pbar)

    #run_time = timeit.default_timer() - StartTime
    #f.write("Run Time: {}\n\n".format(run_time))
