"""
basic trainer
"""
import time
import numpy as np

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable

import utils


__all__ = ["Trainer"]


class Trainer(object):
    """
    trainer for training network, use SGD
    """

    def __init__(self, model, lr_master,
                 train_loader, test_loader,
                 settings, logger=None,
                 optimizer_state=None):
        """
        init trainer
        """

        self.settings = settings

        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr_master = lr_master
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr_master.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weightDecay,
            nesterov=True,
        )
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}

    def reset_optimizer(self, opt_type="SGD"):
        if opt_type == "SGD":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        else:
            assert False, "invalid type: %d" % opt_type

    def reset_model(self, model):
        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr_master.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weightDecay,
            nesterov=True,
        )

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr = self.lr_master.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, images, labels=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        output = self.model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, epoch):
        """
        training
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        iters = len(self.train_loader)
        self.update_lr(epoch)
        self.model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            # if we use multi-gpu, its more efficient to send input
            # to different gpu, instead of send it to the master gpu.

            if self.settings.nGPU == 1:
                images = images.cuda()
            labels = labels.cuda()
            images_var = Variable(images)
            labels_var = Variable(labels)

            output, loss = self.forward(images_var, labels_var)
            self.backward(loss)

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=labels_var,
                loss=loss, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(
                epoch, self.settings.nEpochs, i + 1,
                iters, self.lr_master.lr, data_time, iter_time,
                single_error,
                single_loss, top5error=single5_error,
                mode="Train")

            if self.settings.nEpochs == 1 and i + 1 >= 50:
                print "|===>Program testing for only 50 iterations"
                break

        self.scalar_info['training_top1error'] = top1_error.avg
        self.scalar_info['training_top5error'] = top5_error.avg
        self.scalar_info['training_loss'] = top1_loss.avg

        if self.logger is not None:
            for tag, value in self.scalar_info.items():
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        print "|===>Training Error: %.4f Loss: %.4f, Top5 Error:%.4f" % (
            top1_error.avg, top1_loss.avg, top5_error.avg)

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.test_loader):
            start_time = time.time()
            data_time = start_time - end_time

            labels = labels.cuda()
            labels_var = Variable(labels, volatile=True)
            if self.settings.tenCrop:
                image_size = images.size()
                images = images.view(
                    image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                images_tuple = images.split(image_size[0])
                output = None
                for img in images_tuple:
                    if self.settings.nGPU == 1:
                        img = img.cuda()
                    img_var = Variable(img, volatile=True)
                    temp_output, _ = self.forward(img_var)
                    if output is None:
                        output = temp_output.data
                    else:
                        output = torch.cat((output, temp_output.data))
                single_error, single_loss, single5_error = utils.compute_tencrop(
                    outputs=output, labels=labels_var)
            else:
                if self.settings.nGPU == 1:
                    images = images.cuda()
                images_var = Variable(images, volatile=True)
                output, loss = self.forward(images_var, labels_var)

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output, loss=loss,
                    labels=labels_var, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            utils.print_result(
                epoch, self.settings.nEpochs, i + 1,
                iters, self.lr_master.lr, data_time, iter_time,
                single_error, single_loss,
                top5error=single5_error,
                mode="Test")

            if self.settings.nEpochs == 1 and i + 1 >= 50:
                print "|===>Program testing for only 50 iterations"
                break

        self.scalar_info['testing_top1error'] = top1_error.avg
        self.scalar_info['testing_top5error'] = top5_error.avg
        self.scalar_info['testing_loss'] = top1_loss.avg
        if self.logger is not None:
            for tag, value in self.scalar_info.items():
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1
        print "|===>Testing Error: %.4f Loss: %.4f, Top5 Error: %.4f" % (
            top1_error.avg, top1_loss.avg, top5_error.avg)
        return top1_error.avg, top1_loss.avg, top5_error.avg
