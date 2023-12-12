###########################################################################################
# Machine Learning-based Second-order Analysis of Beam-columns through PINNs
# Developed by Siwei Liu, Liang Chen and Haoyi Zhang
# License: GPL-3.0
###########################################################################################
# ===========================================================================
# Import standard libraries
import timeit, logging, os
# =========================================================================================


class PrintLog:
    @staticmethod
    def Initialize(FileName, ModelName):
        if os.path.exists(FileName + '.rst' + "/" + ModelName + ".log"):
            os.remove(FileName + '.rst' + "/" + ModelName + ".log")

        Logfile = FileName + '.rst' + "/" + ModelName + ".log"
        logging.basicConfig(filename=Logfile, filemode="w", format="%(message)s", level=logging.INFO)
        return

    @staticmethod
    def Print(message):
        print(message)
        logging.info(message)
        return

    def ShowRuntime(self, StartTime):
        tOutPut1 = self.getRunTime(StartTime)
        tOutPut2 = self.EndMessage()
        tOutPut = tOutPut1 + tOutPut2
        return tOutPut

    @staticmethod
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

    @staticmethod
    def EndMessage():
        tOutput = "   " + '\n'
        tOutput += "********************************************************************************" + '\n'
        tOutput += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " + '\n'
        tOutput += "********************************************************************************" + '\n'
        return tOutput

    @staticmethod
    def getRunTime(StartTime):
        Time = timeit.default_timer() - StartTime
        Time = format(Time, "0.2f")
        tOutput = "********************************************************************************" + '\n'
        tOutput += "Run time = " + str(Time) + " s" + '\n'
        tOutput += "********************************************************************************" + '\n'
        return tOutput
