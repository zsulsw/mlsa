from test.LSSI.PINN_Model import BaseNetwork
import timeit
from test.LSSI import IntegralStratergy as IS, AdaoptiveLossWeight, NetworkSL
import os
import torch, math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from Source.File.IO import load_model
from Source.Variables import Model
from Source.Training.Train import train_model
from Source.File import ReadData
import matplotlib


def pre_train():
    path = os.getcwd()
    file_folder = path + "\\Examples\\pre-train_file\\"
    res_folder = path + "\\Examples\\pre-train.rst"
    conf_file = path + "\\Examples\\pre-train.rst\\pre-train.config"
    tar_file = path + "\\Examples\\pre-train.rst\\pre-train.tar"
    file_list = [f for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
    matplotlib.use('TkAgg')
    for i in range(1, 11):
        for j in file_list:
            print("\nLoad Step: {}\n".format(i))
            Model.reset_all()
            ReadData.modelfromJSON(FileName=path + "\\Examples\\pre-train_file\\" + j)
            Model.initialize()
            if os.path.exists(conf_file) and os.path.exists(tar_file):
                model = load_model(res_folder, 'pre-train')
            else:
                model = BaseNetwork(act_fn=[torch.nn.Tanhshrink(), torch.nn.Tanh(), torch.nn.Tanhshrink()],
                                input_size=1, output_size=2, hidden_sizes=[200, 200, 200])
            model.train()
            opt = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.999))
            loss = nn.MSELoss()
            num_epochs = int(Model.Analysis.num_epochs)
            num_sample = Model.Analysis.num_sample
            tol = 0.01 * i ** 2 * Model.Analysis.TOL
            target_LF = Model.Analysis.target_LF
            load_factor = 0.1 * i * target_LF
            # Training loop
            desc = "start training..."
            pbar = tqdm(range(num_epochs), desc=desc)
            train_model(model, res_folder, 'pre-train', load_factor, num_sample, tol, loss, opt, pbar)


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
