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
import timeit, sys, logging, os
# =========================================================================================
# Import internal functions
from Source.Variables import Model
from Source.Utils.PrintLog import PrintLog as pl
from Source.File import ReadData
from Source.NeuralNetwork import NNforBeamColumn
# =========================================================================================
ProgrameName = " MLSA - Machine Learning-based Structural Analysis Module(v1.0.0) "
DeveloperName = " Developed by MLSA Team "
RevisedDate = " Last Revised: May. 21, 2023 "
# =========================================================================================
def Run(argv=''):
    # ----------------------------------------------------------------
    # Initializing
    # ----------------------------------------------------------------
    print("Do you want to start pre-training? (y/n)")
    pretrain = input()
    if pretrain == 'y':
        NNforBeamColumn.pre_train()
    elif pretrain == 'n':
        ReadData.modelfromJSON(FileName=argv)
        Model.initialize()
        pl().Initialize(Model.OutResult.FileName, Model.OutResult.ModelName)
        # logging Logo
        pl.Print(pl.StartMessage(ProgrameName, DeveloperName, RevisedDate))

        #if argv =="":
        # Start Timer
        StartTime = timeit.default_timer()
        # ----------------------------------------------------------------
        # Run Analysis
        # Solver(model).run()
        # ----------------------------------------------------------------
        # Initializing
        # ----------------------------------------------------------------
        NNforBeamColumn.Run()
        pl.Print(pl().ShowRuntime(StartTime))
        # ----------------------------------------------------------------
        # Analysis is completed
        # ----------------------------------------------------------------

# =========================================================================================
if __name__ == '__main__':
    # Initialize the analysis model
    try:
        Run(sys.argv[1])
    except:
        Run()
# =========================================================================================
# END OF PROGRAM
# =========================================================================================