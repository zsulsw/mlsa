###########################################################################################
# Machine Learning-based Second-order Analysis of Beam-columns through PINNs
# Developed by Siwei Liu, Liang Chen and Haoyi Zhang
# License: GPL-3.0
###########################################################################################

import os
import torch
import json
import csv
from Source.NeuralNetwork.BaseModel import BaseNetwork


act_fn_by_name = {
    "Tanh": torch.nn.Tanh,
    "Relu": torch.nn.ReLU,
    'Tanhshrink': torch.nn.Tanhshrink,
    "Softplus": torch.nn.Softplus,
    "Sigmoid": torch.nn.Sigmoid
}


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def load_model(model_path, model_name, net=None, device='cpu'):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        net - (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict.pop("act_fn")
        act_fu = []
        for ii in act_fn_name:
            act_fu.append(act_fn_by_name[ii]())
        net = BaseNetwork(act_fn=act_fu, **config_dict)
    net.load_state_dict(torch.load(model_file, map_location=device))
    return net


def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)


def init_load_deform_res(model_path, model_name):
    # Initialize load-deformation curve data file
    file_name = model_path + "\\" + model_name + "_load_deformation_curve.csv"
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["load factor", "deformation"])


def save_load_deform_res(load_factor, deform_data, model_path, model_name):
    deform_data = deform_data.detach().numpy()
    max_deform = max(abs(deform_data.flatten()))
    # Save load-deformation curve data
    file_name = model_path + "\\" + model_name + "_load_deformation_curve.csv"
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([load_factor, max_deform])


def save_shape_res(x_res, y_res, model_path, model_name, load_factor):
    x_res = x_res.detach().numpy()
    y_res = y_res.detach().numpy()
    # Save deformed shape points in a CSV file
    file_name = model_path + "\\" + model_name + "_deformed_shape_at_load_factor_{:.4f}.csv".format(load_factor)
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["x", "y"])
        # Write the data rows
        for x, y in zip(x_res.flatten(), y_res.flatten()):
            writer.writerow([x, y])
