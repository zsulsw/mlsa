###########################################################################################
# ML - Python-based Cross-platforms Machine Learning Software
#
# Project Leaders :
#   S.W. Liu        -   The Hong Kong Polytechnic University, Hong Kong, China
#
###########################################################################################
# Description:
# =========================================================================================
# Import standard libraries
import timeit
import torch
# =========================================================================================
# Import internal functions
from Source.Variables import Model
from Source.Utils.PrintLog import PrintLog as pl
from Source.File import ReadData
from Source.NeuralNetwork import BeamColumn

# =========================================================================================
ProgrameName = " MLSA - Machine Learning-based Structural Analysis Module(v1.0.0) "
DeveloperName = " Developed by MLSA Team "
RevisedDate = " Last Revised: May. 21, 2023 "
# =========================================================================================


def Run(argv=''):
    # ----------------------------------------------------------------
    # Initializing
    # ----------------------------------------------------------------
    ReadData.modelfromJSON(FileName=argv)
    Model.initialize()
    pl().Initialize(Model.OutResult.FileName, Model.OutResult.ModelName)
    pl.Print(pl.StartMessage(ProgrameName, DeveloperName, RevisedDate))

    # Start Timer
    StartTime = timeit.default_timer()
    # ----------------------------------------------------------------
    # Select Device
    # ----------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------------------
    # Start Analysis
    # ----------------------------------------------------------------
    BeamColumn.Run(device)
    pl.Print(pl().ShowRuntime(StartTime))
    # ----------------------------------------------------------------
    # Analysis Complete
    # ----------------------------------------------------------------

# =========================================================================================


if __name__ == '__main__':
    Run()
# =========================================================================================
# END OF PROGRAM
# =========================================================================================