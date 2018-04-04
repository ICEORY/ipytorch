from ipytorch.options import NetOption

class Option(NetOption):
    def __init__(self):
        super(Option, self).__init__()
        #  ------------ General options ----------------------------------------
        self.dataPath = "/home/dataset"  # path for loading data set
        self.dataset = "cifar10" # options: imagenet | cifar10 | cifar100 | imagenet100 | mnist
        self.nGPU = 1  # number of GPUs to use by default
        self.GPU = 0  # default gpu to use, options: range(nGPU)

        # ------------- Data options -------------------------------------------
        self.nThreads = 16  # number of data loader threads

        # ---------- Optimization options --------------------------------------
        self.nEpochs = 180  # number of total epochs to train
        self.batchSize = 256  # mini-batch size
        self.momentum = 0.9  # momentum
        self.weightDecay = 4e-5  # weight decay 1e-4

        # lr master for optimizer 1 (mask vector d)
        self.lr = 0.1  # initial learning rate 
        self.lrPolicy = "multi_step"  # options: multi_step | linear | exp | const | step
        self.power = 0.98  # power for inv policy (lr_policy)
        self.step = [30, 60, 90, 120, 150]  # step for linear or exp learning rate policy
        self.decayRate = 0.1 # lr decay rate
        self.endlr = -1

        # ---------- Model options ---------------------------------------------
        self.netType = "PreResNet"  # options: ResNet | PreResNet | GreedyNet | NIN | LeNet5 | LeNet500300 | DenseNet_Cifar
        self.experimentID = "compute_madds_0327"
        self.depth = 8  # resnet depth: (n-2)%6==0
        self.nClasses = 10  # number of classes in the dataset
        self.wideFactor = 1  # wide factor for wide-resnet
        self.drawNetwork = False
        # ---------- Resume or Retrain options ---------------------------------------------
        self.resume = None  # "./checkpoint_064.pth"
        self.retrain = None
        self.paramscheck()
