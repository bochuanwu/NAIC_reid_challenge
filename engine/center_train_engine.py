from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import metrics
from . import TrainEngine
from losses import CrossEntropyLoss, TripletLoss, CenterLoss
from utils import AverageMeter, open_specified_layers, open_all_layers
from torch import optim
from itertools import chain


class ImageCenterEngine(TrainEngine):
    r"""Center-loss engine for image-reid.

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


    """

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, weight_c=0.0005, scheduler=None, use_gpu=True,
                 metric='cosine', label_smooth=True, feature_dim=2048, num_features=1, **kwargs):
        super(ImageCenterEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_c = weight_c
        self.num_features = num_features
        self.criterion_t = TripletLoss(margin=margin, metric=metric)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        # multi features for center loss
        self.criterion_c = []
        for i in range(num_features):
            if i < 2:
                criterion_c = CenterLoss(
                    num_classes=self.datamanager.num_train_pids,
                    use_gpu=use_gpu,
                    feat_dim=feature_dim // 2,
                )
            else:
                criterion_c = CenterLoss(
                    num_classes=self.datamanager.num_train_pids,
                    use_gpu=use_gpu,
                    feat_dim=feature_dim,
                )
            self.criterion_c.append(criterion_c)
        self.criterion_c_parameters = list(map(lambda x: x.parameters(), self.criterion_c))
        self.criterion_c_parameters = chain(*self.criterion_c_parameters)
        self.center_optimizer = optim.SGD(self.criterion_c_parameters,
                                          lr=0.5,
                                          weight_decay=5e-4,
                                          momentum=0.9)

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses_c = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()
            self.center_optimizer.zero_grad()
            outputs, features = self.model(imgs)
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss_c = 0.
            for feature, criterion_c in zip(features, self.criterion_c):
                loss_c += self._compute_loss(criterion_c, feature, pids)
            loss = self.weight_t * loss_t + self.weight_x * loss_x + self.weight_c * loss_c
            loss.backward()

            for param in  self.criterion_c_parameters:
                param.grad.data *= (1. / self.weight_c)
            self.optimizer.step()
            self.center_optimizer.step()

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))
            losses_c.update(loss_c.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                      'Loss_c {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_t=losses_t,
                    loss_x=losses_x,
                    loss_c=losses_c,
                    acc=accs,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss_t', losses_t.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x', losses_x.avg, n_iter)
                self.writer.add_scalar('Train/loss_c', losses_c.avg, n_iter)
                self.writer.add_scalar('Train/Acc', accs.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
