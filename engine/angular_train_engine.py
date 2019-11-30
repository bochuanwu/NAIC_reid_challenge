from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import metrics
from losses.angular_penalty_softmax_loss import AngularPenaltySMLoss
from . import TrainEngine
from losses import TripletLoss, ArcMarginProduct
from utils import AverageMeter, open_specified_layers, open_all_layers

try:
    from apex import amp
except:
    pass


class ImageAngularEngine(TrainEngine):
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
    """

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_gpu=True, label_smooth=True,
                 metric='cosine'):
        super(ImageAngularEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.metric = metric
        self.criterion_t = TripletLoss(margin=margin, metric=metric)
        self.criterion_x = ArcMarginProduct(num_classes=self.datamanager.num_train_pids,
                                            use_gpu=self.use_gpu,
                                            label_smooth=label_smooth)

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()

        # if epoch == 30:
        #     self.criterion_t = TripletLoss(margin=0.3, metric=self.metric)
        # elif epoch == 60:
        #     self.criterion_t = TripletLoss(margin=0.5, metric=self.metric)
        # elif epoch == 90:
        #     self.criterion_t = TripletLoss(margin=0.7, metric=self.metric)

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
            outputs, features = self.model(imgs)
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss = self.weight_t * loss_t + self.weight_x * loss_x

            if self.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))
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
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_t=losses_t,
                    loss_x=losses_x,
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
                self.writer.add_scalar('Train/Acc', accs.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
