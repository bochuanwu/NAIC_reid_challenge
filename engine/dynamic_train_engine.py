from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch
import math
import metrics
from . import TrainEngine
from losses import CrossEntropyLoss, TripletLoss, RankedLoss, CRankedLoss, MultiSimilarityLoss
from utils import AverageMeter, open_specified_layers, open_all_layers

try:
    from apex import amp
except:
    pass


class ImageDynamicEngine(TrainEngine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, margin=0.3, scheduler=None, use_gpu=True,
                 metric='euclidean', label_smooth=True, T=2, apex=False):
        super(ImageDynamicEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu, apex)

        self.metric = metric
        self.criterion_t = TripletLoss(margin=margin, metric=metric)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.n_iter = 0
        self.losses_x_bank = []  # record cross entropy loss value
        self.losses_t_bank = []  # record triplet loss value
        self.w_t_iter = 1.
        self.w_x_iter = 1.
        self.T = T

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10, **kwargs):
        weighting_losses_t = AverageMeter()
        weighting_losses_x = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_t = []
        losses_x = []

        self.model.train()

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):

            self.n_iter += 1
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()

            outputs, features = self.model(imgs)
            loss_t = self._compute_loss(self.criterion_t, features, pids, kwargs['loss_reduce'])
            loss_x = self._compute_loss(self.criterion_x, outputs, pids, kwargs['loss_reduce'])

            if epoch > 1:
                lambda_t_iter = self.losses_t_bank[-1] / (self.losses_t_bank[-2])
                lambda_x_iter = self.losses_x_bank[-1] / (self.losses_x_bank[-2])
                w_t_iter = math.exp(lambda_t_iter / self.T)
                w_x_iter = math.exp(lambda_x_iter / self.T)
                self.w_t_iter = 2 * w_t_iter / (w_t_iter + w_x_iter)
                self.w_x_iter = 2 * w_x_iter / (w_t_iter + w_x_iter)

            losses_t.append(loss_t.item())
            losses_x.append(loss_x.item())

            loss_x = loss_x * self.w_x_iter
            loss_t = loss_t * self.w_t_iter

            weighting_losses_t.update(loss_t.item(), pids.size(0))
            weighting_losses_x.update(loss_x.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            loss = loss_x + loss_t

            if self.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                      'Lr {lr:.8f}\t'
                      'eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_t=weighting_losses_t,
                    loss_x=weighting_losses_x,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            if self.writer is not None:
                self.writer.add_scalar('Train/Time', batch_time.avg, self.n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, self.n_iter)
                self.writer.add_scalar('Train/Loss_t', weighting_losses_t.avg, self.n_iter)
                self.writer.add_scalar('Train/Loss_x', weighting_losses_x.avg, self.n_iter)
                self.writer.add_scalar('Train/Acc', accs.avg, self.n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], self.n_iter)
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        print('Epoch: [{0}/{1}]\t'
              'Epoch Time {batch_time.sum:.3f} s\t '
              'Lr {lr:.8f}\t\n'.format(
            epoch + 1, max_epoch,
            batch_time=batch_time,
            lr=self.optimizer.param_groups[0]['lr']))

        self.losses_x_bank.append(sum(losses_x) * 1. / len(losses_x))
        self.losses_t_bank.append(sum(losses_t) * 1. / len(losses_t))
