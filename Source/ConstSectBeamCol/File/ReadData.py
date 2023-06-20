#############################################################################
# MLSA - Machine Learning-based Structural Analysis
#############################################################################
# ===========================================================================
# Import standard libraries
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import json

# =========================================================================================
# Import internal functions
from Source.ConstSectBeamCol.Variables import Model


def Getfilename():
    print("The following files are found. Please enter a file name to execute:")
    JSONScript = os.getcwd() + "\\examples\\"
    Model.OutResult.Folder = os.getcwd() + "\\examples\\"
    files = [f for f in listdir(JSONScript) if isfile(join(JSONScript, f))]
    print(files)
    FileName = input(">>")
    while not (FileName in files):
        print("File is not found, please enter the file name from the list")
        FileName = input(">>")
    Model.OutResult.ModelName = FileName
    OutFolder = os.path.join(JSONScript, FileName + '.rst')
    if not os.path.exists(OutFolder):
        os.makedirs(OutFolder)
    Model.OutResult.FileName = JSONScript + FileName
    return JSONScript + FileName


def Savefileinfo(FileName):
    JSONScript = os.getcwd() + "\\examples\\"
    Model.OutResult.Folder = os.getcwd() + "\\"
    FileName = FileName.split("\\")
    FileName = FileName[len(FileName) - 1]
    Model.OutResult.ModelName = FileName
    OutFolder = os.path.join(JSONScript, FileName + '.rst')
    if not os.path.exists(OutFolder):
        os.makedirs(OutFolder)
    Model.OutResult.FileName = JSONScript + FileName


# Read data file from JSON format
def ReadJSON():
    FileName = Getfilename()
    f = open(FileName, 'r')
    return json.loads(f.read())


# Load Data from JSON fomat to Mastan Model
def LoadDataToModel(DataIn):
    Model.Information.ReadModelGenlInfo(np.array(DataIn["INFORMATION"]))
    Model.Node.ReadNode(Model.Node, np.array(DataIn["NODE"]))
    Model.Member.ReadMember(np.array(DataIn["MEMBER"]))
    Model.Material.ReadMat(np.array(DataIn["MATERIAL"]))
    Model.Section.ReadSect(np.array(DataIn["SECTION"]))
    Model.Boundary.ReadBoun(np.array(DataIn["BOUNDARY"]))
    Model.MemberUDL.ReadMUDL(np.array(DataIn["MEMBERUDL"]))
    try:
        Model.Coupling.ReadCoupl(DataIn["COUPLING"])
    except:
        pass
    Model.JointLoad.ReadJNTL(np.array(DataIn["JOINTLOAD"]))
    Model.Analysis.ReadAna(np.array(DataIn["ANALYSIS"]))
    return


def modelfromJSON(FileName=''):
    if FileName == "":
        DataIn = ReadJSON()
    else:
        print(FileName)
        f = open(FileName, 'r')
        Savefileinfo(FileName)
        DataIn = json.loads(f.read())
    LoadDataToModel(DataIn)
    return


def ReadJSON_GUI(FileName):
    f = open(FileName, 'r')
    return json.loads(f.read())
