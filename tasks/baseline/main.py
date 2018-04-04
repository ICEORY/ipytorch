import datetime
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from termcolor import colored
from torch.autograd import Variable

import ipytorch.models as md
import ipytorch.utils as utils
import ipytorch.visualization as vs
from ipytorch.checkpoint import CheckPoint
from ipytorch.dataloader import DataLoader
from ipytorch.trainer import Trainer
# option file should be modified according to your expriment
from options import Option

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 3"


class ExperimentDesign:
    def __init__(self, options=None):
        self.settings = options or Option()
        self.checkpoint = None
        self.train_loader = None
        self.test_loader = None
        self.model = None

        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0
        self.test_input = None

        self.visualize = vs.Visualization(self.settings.save_path)
        self.logger = vs.Logger(self.settings.save_path)

        self.prepare()

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._set_trainer()
        self._draw_net()

    def _set_gpu(self):
        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        torch.cuda.set_device(self.settings.GPU)
        cudnn.benchmark = True

    def _set_dataloader(self):
        # create data loader
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=self.settings.batchSize,
                                 data_path=self.settings.dataPath,
                                 n_threads=self.settings.nThreads,
                                 ten_crop=self.settings.tenCrop)

        self.train_loader, self.test_loader = data_loader.getloader()

    def _set_checkpoint(self):

        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path)

        if self.settings.retrain is not None:
            model_state = self.checkpoint.load_model(self.settings.retrain)
            self.model = self.checkpoint.load_state(self.model, model_state)

        if self.settings.resume is not None:
            model_state, optimizer_state, epoch = self.checkpoint.load_checkpoint(
                self.settings.resume)
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.start_epoch = epoch
            self.optimizer_state = optimizer_state

    def _set_model(self):
        if self.settings.dataset in ["cifar10", "cifar100"]:
            self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
            if self.settings.netType == "PreResNet":
                self.model = md.official.PreResNet(depth=self.settings.depth,
                                                   num_classes=self.settings.nClasses,
                                                   wide_factor=self.settings.wideFactor)

            elif self.settings.netType == "PreResNet_Test":
                self.model = md.PreResNet_Test(depth=self.settings.depth,
                                               num_classes=self.settings.nClasses,
                                               wide_factor=self.settings.wideFactor,
                                               max_conv=10)

            elif self.settings.netType == "DenseNet_Cifar":
                self.model = md.official.DenseNet_Cifar(depth=self.settings.depth,
                                                        num_classes=self.settings.nClasses,
                                                        reduction=1.0,
                                                        bottleneck=False)
            elif self.settings.netType == "NetworkInNetwork":
                self.model = md.NetworkInNetwork()

            elif self.settings.netType == "VGG":
                self.model = md.VGG_CIFAR(
                    self.settings.depth, num_classes=self.settings.nClasses)
            else:
                assert False, "use %s data while network is %s" % (
                    self.settings.dataset, self.settings.netType)

        elif self.settings.dataset == "mnist":
            self.test_input = Variable(torch.randn(1, 1, 28, 28).cuda())
            if self.settings.netType == "LeNet5":
                self.model = md.LeNet5()
            elif self.settings.netType == "LeNet500300":
                self.model = md.LeNet500300()
            else:
                assert False, "use mnist data while network is:" + self.settings.netType

        elif self.settings.dataset in ["imagenet", "imagenet100"]:
            if self.settings.netType == "resnet18":
                self.model = md.resnet18()
            elif self.settings.netType == "resnet34":
                self.model = md.resnet34()
            elif self.settings.netType == "resnet50":
                self.model = md.resnet50()
            elif self.settings.netType == "resnet101":
                self.model = md.resnet101()
            elif self.settings.netType == "resnet152":
                self.model = md.resnet152()
            elif self.settings.netType == "VGG":
                self.model = md.VGG(
                    depth=self.settings.depth, bn_flag=False, num_classes=self.settings.nClasses)
            elif self.settings.netType == "VGG_GAP":
                self.model = md.VGG_GAP(
                    depth=self.settings.depth, bn_flag=False, num_classes=self.settings.nClasses)
            elif self.settings.netType == "Inception3":
                self.model = md.Inception3(num_classes=self.settings.nClasses)
            elif self.settings.netType == "MobileNet_v2":
                self.model = md.MobileNet_v2(
                    num_classes=self.settings.nClasses,
                    )# wide_scale=1.4)
            else:
                assert False, "use %s data while network is%s" % (
                    self.settings.dataset, self.settings.netType)

            if self.settings.netType in ["InceptionResNetV2", "Inception3"]:
                self.test_input = Variable(torch.randn(1, 3, 299, 299).cuda())
            else:
                self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
        else:
            assert False, "unsupport data set: " + self.settings.dataset

    def _set_trainer(self):
        # set lr master
        lr_master = utils.LRPolicy(self.settings.lr,
                                   self.settings.nEpochs,
                                   self.settings.lrPolicy)
        params_dict = {
            'power': self.settings.power,
            'step': self.settings.step,
            'end_lr': self.settings.endlr,
            'decay_rate': self.settings.decayRate
        }

        lr_master.set_params(params_dict=params_dict)
        # set trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr_master=lr_master,
            settings=self.settings,
            logger=self.logger,
            optimizer_state=self.optimizer_state
        )
        # self.trainer.reset_optimizer(opt_type="RMSProp")

    def _draw_net(self):
        if self.settings.drawNetwork:
            rand_output, _ = self.trainer.forward(self.test_input)
            self.visualize.save_network(rand_output)
            self.visualize.write_settings(self.settings)

    def _model_analyse(self, model):
        # analyse model
        model_analyse = utils.ModelAnalyse(model, self.visualize)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        print "zero rate is: ", zero_rate

        # save analyse result to file
        self.visualize.write_readme(
            "Number of parameters is: %d, number of zeros is: %d, zero rate is: %f" % (params_num, zero_num, zero_rate))

        # model_analyse.flops_compute(self.test_input)
        model_analyse.madds_compute(self.test_input)

    def run(self, run_count=0):
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()
        # self.trainer.test(0)
        # assert False
        self._model_analyse(self.model)
        assert False
        for epoch in range(self.start_epoch, self.settings.nEpochs):
            self.start_epoch = 0
            # training and testing
            train_error, train_loss, train5_error = self.trainer.train(
                epoch=epoch)
            test_error, test_loss, test5_error = self.trainer.test(
                epoch=epoch)
            # self.trainer.model.apply(utils.SVB)
            # self.trainer.model.apply(utils.BBN)

            # write and print result
            log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                epoch, train_error, train_loss, test_error,
                test_loss, train5_error, test5_error)

            self.visualize.write_log(log_str)
            best_flag = False
            if best_top1 >= test_error:
                best_top1 = test_error
                best_top5 = test5_error
                best_flag = True

                print colored("# %d ==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (
                    run_count, best_top1, best_top5), "red")
            else:
                print colored("# %d ==>Best Result is: Top1 Error: %f, Top5 Error: %f\n" % (
                    run_count, best_top1, best_top5), "blue")
            if self.settings.dataset in ["imagenet", "imagenet100"]:
                self.checkpoint.save_checkpoint(
                    self.model, self.trainer.optimizer, epoch, epoch)
            else:
                self.checkpoint.save_checkpoint(
                    self.model, self.trainer.optimizer, epoch)

            if best_flag:
                self.checkpoint.save_model(self.model, best_flag=best_flag)

            if (epoch + 1) % self.settings.drawInterval == 0:
                self.visualize.draw_curves()

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Running Time is: " + \
            str(datetime.timedelta(seconds=time_interval)) + "\n"
        print(t_string)

        self.visualize.write_settings(self.settings)
        # save experimental results
        self.visualize.write_readme(
            "Best Result of all is: Top1 Error: %f, Top5 Error: %f\n" % (best_top1, best_top5))

        self.visualize.draw_curves()

        # analyse model
        self._model_analyse(self.model)
        return best_top1
# ---------------------------------------------------------------------------------------------


def main():
    '''options = BasicOption()
    depth_list = [116, 56, 44, 32, 20]
    options.netType = "PreResNet"
    options.nGPU = 1
    options.GPU = 1
    acc_all = []
    log_file = "cifar10_preresnet_exp_repo_001.log"
    for depth in depth_list:
        options.depth = depth
        options.extra_checking()
        options.paramscheck()
        exp = ExperimentDesign(options)
        acc = exp.run()
        acc_all.append(acc)
        txt_file = open(log_file, 'a+')
        txt_file.write("%s%d-%f\n"%(options.netType, options.depth, acc))
        txt_file.close()
    txt_file = open(log_file, 'a+')
    txt_file.write("============summary=================\n")
    txt_file.write("%s\n"%(str(acc_all)))
    txt_file.close()'''
    experiment = ExperimentDesign()
    experiment.run()
    # experiment.analyse()


if __name__ == '__main__':
    main()
