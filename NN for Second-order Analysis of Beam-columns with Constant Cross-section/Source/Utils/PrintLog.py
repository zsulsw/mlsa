#############################################################################
# MLSA - Machine Learning-based Structural Analysis

#############################################################################
# Description:
# ===========================================================================
# Import standard libraries
#import numpy as np
#import math
import timeit, sys, logging, os
# Import internal functions
# =========================================================================================

class PrintLog:
    #StartTime = 0.0

    def Initialize(self, FileName, ModelName):
        if os.path.exists(FileName + '.rst' + "/" + ModelName + ".log"):
            os.remove(FileName + '.rst' + "/" + ModelName + ".log")

        Logfile = FileName + '.rst' + "/" + ModelName + ".log"
        #logging.basicConfig(filename=Logfile,filemode="w",format="[%(asctime)s]:\t%(message)s",datefmt="%H:%M:%S",level=logging.INFO)
        logging.basicConfig(filename=Logfile, filemode="w", format="%(message)s", level=logging.INFO)

        #self.StartTime = timeit.default_timer()
        return

    def Print(message):
        print(message)
        logging.info(message)
        return

    def ShowRuntime(self, StartTime):
        tOutPut1 = self.getRunTime(StartTime)
        tOutPut2 = self.EndMessage()
        tOutPut = tOutPut1 + tOutPut2
        return tOutPut

    def StartMessage(tName, tAuthors, tRevisedDate):
        # tOutput = "   " + '\n'
        tOutput = "********************************************************************************" + '\n'
        tOutput += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MLSA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " + '\n'
        tOutput += "********************************************************************************" + '\n'
        tOutput += "   " + '\n'
        tOutput += "Programe Name: " + tName + '\n'
        tOutput += "Authors: " + tAuthors + '\n'
        tOutput += "Last Revised: " + tRevisedDate + '\n'
        tOutput += "Note: MASTAN3 - Python-based Cross-platforms Frame Analysis Software"
        return tOutput

    def EndMessage(self):
        tOutput = "   " + '\n'
        tOutput += "********************************************************************************" + '\n'
        tOutput += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " + '\n'
        tOutput += "********************************************************************************" + '\n'
        return tOutput

    def getRunTime(self, StartTime):
        Time = timeit.default_timer() - StartTime
        Time = format(Time, "0.2f")
        tOutput = "********************************************************************************" + '\n'
        tOutput += "Run time = " + str(Time) + " s" + '\n'
        tOutput += "********************************************************************************" + '\n'
        return tOutput

