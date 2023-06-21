#############################################################################
# MLSA - Machine Learning-based Structural Analysis
#############################################################################
# ===========================================================================
# Import standard libraries
import numpy as np
# ===========================================================================


def initialize():
    # Member Geometry
    Member.Initialize(Member.Count)
    return


class Information:
    Version = " "
    EDate = " "
    Description = " "

    @classmethod
    def reset(cls):
        cls.Version = " "
        cls.EDate = " "
        cls.Description = " "

    def ReadModelGenlInfo(ModelGenlInfo):
        Information.Version = ModelGenlInfo[0, 1]
        Information.EDate = ModelGenlInfo[1, 1]
        Information.Description = ModelGenlInfo[2, 1]
        return


class Node:
    Count = 0
    ID = []
    X = {}
    Y = {}
    Z = {}
    thX = {}
    thY = {}
    thZ = {}
    thB = {}
    X0 = {}
    Y0 = {}
    Z0 = {}
    thX0 = {}
    thY0 = {}
    thZ0 = {}
    thB0 = {}
    Fg = {}
    CurU = {}
    CurF = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.ID = []
        cls.X = {}
        cls.Y = {}
        cls.Z = {}
        cls.thX = {}
        cls.thY = {}
        cls.thZ = {}
        cls.thB = {}
        cls.X0 = {}
        cls.Y0 = {}
        cls.Z0 = {}
        cls.thX0 = {}
        cls.thY0 = {}
        cls.thZ0 = {}
        cls.thB0 = {}
        cls.Fg = {}
        cls.CurU = {}
        cls.CurF = {}

    def __init__(self):
        self.Count = 0
        self.ID = []
        self.X = {}
        self.Y = {}
        self.Z = {}
        self.thX = {}
        self.thY = {}
        self.thZ = {}
        self.thB = {}
        self.X0 = {}
        self.Y0 = {}
        self.Z0 = {}
        self.thX0 = {}
        self.thY0 = {}
        self.thZ0 = {}
        self.thB0 = {}
        self.Fg = {}
        self.CurU = {}
        self.CurF = {}

    def ReadNode(self, NodeInfo) -> object:
        self.Count = len(NodeInfo)
        self.ID = dict(zip(NodeInfo[:, 0], np.arange(Node.Count)))
        self.X = dict(zip(NodeInfo[:, 0], NodeInfo[:, 1]))
        self.Y = dict(zip(NodeInfo[:, 0], NodeInfo[:, 2]))
        self.Z = dict(zip(NodeInfo[:, 0], NodeInfo[:, 3]))
        self.thX = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thY = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thZ = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thB = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.X0 = dict(zip(NodeInfo[:, 0], NodeInfo[:, 1]))
        self.Y0 = dict(zip(NodeInfo[:, 0], NodeInfo[:, 2]))
        self.Z0 = dict(zip(NodeInfo[:, 0], NodeInfo[:, 3]))
        self.thX0 = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thY0 = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thZ0 = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.thB0 = dict(zip(NodeInfo[:, 0], np.zeros(Node.Count)))
        self.Fg = np.zeros(Node.Count * 7)
        self.CurU = np.zeros(Node.Count * 7)
        self.CurF = np.zeros(Node.Count * 7)
        return


class Member:
    Count = 0
    ID = []
    SectID = []
    I = {}
    J = {}
    L0 = {}
    Beta = {}
    Imperfection={}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.ID = []
        cls.SectID = []
        cls.I = {}
        cls.J = {}
        cls.L0 = {}
        cls.Beta = {}
        cls.Imperfection={}

    def Initialize(tMemCount):
        #
        Member.L = np.zeros(tMemCount)
        # Member.L0 = np.zeros(tMemCount)
        Member.x1 = np.zeros(tMemCount)
        Member.y1 = np.zeros(tMemCount)
        Member.z1 = np.zeros(tMemCount)
        Member.Fx1 = np.zeros(tMemCount)
        Member.Fy1 = np.zeros(tMemCount)
        Member.Fz1 = np.zeros(tMemCount)
        Member.Mx1 = np.zeros(tMemCount)
        Member.My1 = np.zeros(tMemCount)
        Member.Mz1 = np.zeros(tMemCount)
        Member.Mb1 = np.zeros(tMemCount)
        Member.thx1 = np.zeros(tMemCount)
        Member.thy1 = np.zeros(tMemCount)
        Member.thz1 = np.zeros(tMemCount)
        Member.thb1 = np.zeros(tMemCount)
        Member.x2 = np.zeros(tMemCount)
        Member.y2 = np.zeros(tMemCount)
        Member.z2 = np.zeros(tMemCount)
        Member.Fx2 = np.zeros(tMemCount)
        Member.Fy2 = np.zeros(tMemCount)
        Member.Fz2 = np.zeros(tMemCount)
        Member.Mx2 = np.zeros(tMemCount)
        Member.My2 = np.zeros(tMemCount)
        Member.Mz2 = np.zeros(tMemCount)
        Member.Mb2 = np.zeros(tMemCount)
        Member.thx2 = np.zeros(tMemCount)
        Member.thy2 = np.zeros(tMemCount)
        Member.thz2 = np.zeros(tMemCount)
        Member.thb2 = np.zeros(tMemCount)
        Member.delthx1 = np.zeros(tMemCount)
        Member.delthy1 = np.zeros(tMemCount)
        Member.delthz1 = np.zeros(tMemCount)
        Member.delthx2 = np.zeros(tMemCount)
        Member.delthy2 = np.zeros(tMemCount)
        Member.delthz2 = np.zeros(tMemCount)
        Member.EleMtxL = np.zeros((Member.Count * 3, 3))
        Member.EleMtxL0 = np.zeros((Member.Count * 3, 3))
        Member.EleMtxK = np.zeros((Member.Count * 14, 14))
        # Initialize member length
        for ii in Member.ID:
            tI = Member.I[ii]
            tJ = Member.J[ii]
            X1 = Node.X0[tI]
            Y1 = Node.Y0[tI]
            Z1 = Node.Z0[tI]
            X2 = Node.X0[tJ]
            Y2 = Node.Y0[tJ]
            Z2 = Node.Z0[tJ]
            Member.L0[ii] = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)

    def ReadMember(MembInfo):
        Member.Count = len(MembInfo)
        Member.ID = dict(zip(MembInfo[:, 0], np.arange(Member.Count)))
        Member.SectID = dict(zip(MembInfo[:, 0], MembInfo[:, 1]))
        Member.I = dict(zip(MembInfo[:, 0], MembInfo[:, 2]))
        Member.J = dict(zip(MembInfo[:, 0], MembInfo[:, 3]))
        Member.Beta = dict(zip(MembInfo[:, 0], MembInfo[:, 4]))
        Member.Imperfection = dict(zip(MembInfo[:, 0], MembInfo[:, 5]))
        Member.L0 = dict(zip(MembInfo[:, 0], np.zeros(Member.Count)))
        Member.Initialize(Member.Count)
        return


class Material:
    Count = 0
    ID = []
    E = {}
    G = {}
    Fy = {}
    Dens = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.ID = []
        cls.E = {}
        cls.G = {}
        cls.Fy = {}
        cls.Dens = {}

    def ReadMat(MatInfo):
        Material.Count = len(MatInfo)
        Material.ID = dict(zip(MatInfo[:, 0], np.arange(Material.Count)))
        Material.E = dict(zip(MatInfo[:, 0], MatInfo[:, 1]))
        Material.G = dict(zip(MatInfo[:, 0], MatInfo[:, 2]))
        Material.Fy = dict(zip(MatInfo[:, 0], MatInfo[:, 3]))
        Material.Dens = dict(zip(MatInfo[:, 0], MatInfo[:, 4]))
        return


class Section:
    Count = 0
    ID = []
    ElementType = 2
    MatID = []
    A = {}
    Iy = {}
    Iz = {}
    J = {}
    Cw = {}
    yc = {}
    zc = {}
    ky = {}
    kz = {}
    betay = {}
    betaz = {}
    betaw = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.ID = []
        cls.ElementType = 2
        cls.MatID = []
        cls.A = {}
        cls.Iy = {}
        cls.Iz = {}
        cls.J = {}
        cls.Cw = {}
        cls.yc = {}
        cls.zc = {}
        cls.ky = {}
        cls.kz = {}
        cls.betay = {}
        cls.betaz = {}
        cls.betaw = {}

    # Read Section Dimensions Information
    def ReadSect(SectInfo):
        Section.Count = len(SectInfo)
        Section.ID = dict(zip(SectInfo[:, 0], np.arange(Section.Count)))
        Section.MatID = dict(zip(SectInfo[:, 0], SectInfo[:, 1]))
        Section.ElementType = dict(zip(SectInfo[:, 0], SectInfo[:, 2]))
        Section.A = dict(zip(SectInfo[:, 0], SectInfo[:, 3]))
        Section.Iy = dict(zip(SectInfo[:, 0], SectInfo[:, 4]))
        Section.Iz = dict(zip(SectInfo[:, 0], SectInfo[:, 5]))
        Section.J = dict(zip(SectInfo[:, 0], SectInfo[:, 6]))
        Section.Cw = dict(zip(SectInfo[:, 0], SectInfo[:, 7]))
        Section.yc = dict(zip(SectInfo[:, 0], SectInfo[:, 8]))
        Section.zc = dict(zip(SectInfo[:, 0], SectInfo[:, 9]))
        Section.ky = dict(zip(SectInfo[:, 0], SectInfo[:, 10]))
        Section.kz = dict(zip(SectInfo[:, 0], SectInfo[:, 11]))
        Section.betay = dict(zip(SectInfo[:, 0], SectInfo[:, 12]))
        Section.betaz = dict(zip(SectInfo[:, 0], SectInfo[:, 13]))
        Section.betaw = dict(zip(SectInfo[:, 0], SectInfo[:, 14]))
        return


class Boundary:
    Count = 0
    NodeID = []
    UX = {}
    UY = {}
    UZ = {}
    RX = {}
    RY = {}
    RZ = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.NodeID = []
        cls.UX = {}
        cls.UY = {}
        cls.UZ = {}
        cls.RX = {}
        cls.RY = {}
        cls.RZ = {}

    def ReadBoun(BounInfo):
        Boundary.Count = len(BounInfo)
        Boundary.NodeID = dict(zip(BounInfo[:, 0], np.arange(Boundary.Count)))
        Boundary.UX = dict(zip(BounInfo[:, 0], BounInfo[:, 1]))
        Boundary.UY = dict(zip(BounInfo[:, 0], BounInfo[:, 2]))
        Boundary.UZ = dict(zip(BounInfo[:, 0], BounInfo[:, 3]))
        Boundary.RX = dict(zip(BounInfo[:, 0], BounInfo[:, 4]))
        Boundary.RY = dict(zip(BounInfo[:, 0], BounInfo[:, 5]))
        Boundary.RZ = dict(zip(BounInfo[:, 0], BounInfo[:, 6]))
        return


class Coupling:
    Count = 0
    ID = {}
    Master = {}
    Slave = {}
    UX = {}
    UY = {}
    UZ = {}
    RX = {}
    RY = {}
    RZ = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.ID = {}
        cls.Master = {}
        cls.Slave = {}
        cls.UX = {}
        cls.UY = {}
        cls.UZ = {}
        cls.RX = {}
        cls.RY = {}
        cls.RZ = {}

    def ReadCoupl(CouplInfo):
        tMaster = []
        tSlave = []
        tUX = []
        tUY = []
        tUZ = []
        tRX = []
        tRY = []
        tRZ = []
        for ii in CouplInfo:
            for jj in ii[1:]:
                tMaster.append(ii[0])
                tSlave.append(jj[0])
                tUX.append(jj[1])
                tUY.append(jj[2])
                tUZ.append(jj[3])
                tRX.append(jj[4])
                tRY.append(jj[5])
                tRZ.append(jj[6])
        Coupling.Count = len(tMaster)
        Coupling.ID = dict(enumerate(range(Coupling.Count)))
        Coupling.Master = dict(enumerate(tMaster))
        Coupling.Slave = dict(enumerate(tSlave))
        Coupling.UX = dict(enumerate(tUX))
        Coupling.UY = dict(enumerate(tUY))
        Coupling.UZ = dict(enumerate(tUZ))
        Coupling.RX = dict(enumerate(tRX))
        Coupling.RY = dict(enumerate(tRY))
        Coupling.RZ = dict(enumerate(tRZ))
        return


class JointLoad:
    Count = 0
    NodeID = []
    FX = {}
    FY = {}
    FZ = {}
    MX = {}
    MY = {}
    MZ = {}
    Yp = {}
    Zp = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.NodeID = []
        cls.FX = {}
        cls.FY = {}
        cls.FZ = {}
        cls.MX = {}
        cls.MY = {}
        cls.MZ = {}
        cls.Yp = {}
        cls.Zp = {}


    def ReadJNTL(JNTLInfo):
        JointLoad.Count = len(JNTLInfo)
        JointLoad.NodeID = dict(zip(JNTLInfo[:, 0], np.arange(JointLoad.Count)))
        JointLoad.FX = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 1]))
        JointLoad.FY = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 2]))
        JointLoad.FZ = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 3]))
        JointLoad.MX = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 4]))
        JointLoad.MY = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 5]))
        JointLoad.MZ = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 6]))
        JointLoad.Yp = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 7]))
        JointLoad.Zp = dict(zip(JNTLInfo[:, 0], JNTLInfo[:, 8]))
        return


class MemberUDL:
    Count = 0
    MemberID = []
    QX1 = {}
    QY1 = {}
    QZ1 = {}
    QX2 = {}
    QY2 = {}
    QZ2 = {}

    @classmethod
    def reset(cls):
        cls.Count = 0
        cls.MemberID = []
        cls.QX1 = {}
        cls.QY1 = {}
        cls.QZ1 = {}
        cls.QX2 = {}
        cls.QY2 = {}
        cls.QZ2 = {}


    def ReadMUDL(MUDLInfo):
        MemberUDL.Count = len(MUDLInfo)
        MemberUDL.MemberID = dict(zip(MUDLInfo[:, 0], np.arange(MemberUDL.Count)))
        MemberUDL.QX1 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 1]))
        MemberUDL.QY1 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 2]))
        MemberUDL.QZ1 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 3]))
        MemberUDL.QX2 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 4]))
        MemberUDL.QY2 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 5]))
        MemberUDL.QZ2 = dict(zip(MUDLInfo[:, 0], MUDLInfo[:, 6]))
        return


class Analysis:
    UseTransferLearning = ""
    num_sample = 0.0
    num_epochs = 0.0
    TOL = 0.0
    target_LF = 0.0
    load_step = 0.0

    @classmethod
    def reset(cls):
        cls.UseTransferLearning = ""
        cls.num_sample = 0.0
        cls.num_epochs = 0.0
        cls.TOL = 0.0
        cls.target_LF = 0.0
        cls.load_step = 0.0

    # Read Analysis Information
    def ReadAna(tAanalInfo):
        # print("tAanalInfo = ", tAanalInfo)
        AnalInfo =dict(zip(tAanalInfo[:, 0], tAanalInfo[:, 1]))
        Analysis.UseTransferLearning = AnalInfo.get('UseTransferLearning', "Y")
        Analysis.num_sample = float(AnalInfo.get('num_sample', 10))
        Analysis.num_epochs = int(AnalInfo.get('num_epochs', 999))
        Analysis.TOL = float(AnalInfo.get('TOL', 0.001))
        Analysis.target_LF = int(AnalInfo.get('target_LF', 1))
        Analysis.load_step = float(AnalInfo.get('load_step', 10))


class OutResult:
    FileName = ""
    Folder = ""
    ModelName = ""

    @classmethod
    def reset(cls):
        cls.FileName = ""
        cls.Folder = ""
        cls.ModelName = ""


def reset_all():
    Information.reset()
    Node.reset()
    Member.reset()
    Material.reset()
    Section.reset()
    Boundary.reset()
    Coupling.reset()
    JointLoad.reset()
    MemberUDL.reset()
    Analysis.reset()
    OutResult.reset()
