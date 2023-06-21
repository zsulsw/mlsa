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
import sys
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
    # Start Analysis
    # ----------------------------------------------------------------
    BeamColumn.Run()
    pl.Print(pl().ShowRuntime(StartTime))
    # ----------------------------------------------------------------
    # Analysis Complete
    # ----------------------------------------------------------------

# =========================================================================================


if __name__ == '__main__':
    try:
        Run(sys.argv[1])
    except:
        Run()
# =========================================================================================
# END OF PROGRAM
# =========================================================================================