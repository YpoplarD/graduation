from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F

class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2, loss3, loss4, loss5, prec1 = self._forward(inputs, targets)
#===================================================================================
            loss = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            
            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            #torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5],[torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda()]) 
            total_loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 6
            total_loss.backward()


            
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        #print(f"模型类型: {'ViT' if hasattr(self.model.module, 'transformer') else 'CLIP'}")
        

        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()
##########################################
        if hasattr(self.model.module, 'dino_model'):
            if isinstance(outputs, tuple):
            # 训练模式：DINO返回 (features, logits)
                features = outputs[0]
                logits = outputs[1] if len(outputs) > 1 else None
            else:
            # 测试模式或其他情况
                features = outputs
                logits = self.model.module.classifier(features) if hasattr(self.model.module, 'classifier') else None
        
        # 计算损失和精度
            if logits is not None:
                loss = self.criterion(logits, targets)
                prec = accuracy(logits.data, targets.data)[0]
            else:
                loss = torch.tensor(0., device=targets.device)
                prec = 0
        
        # 保持原接口返回6个相同的loss和1个精度值
            return (loss,) * 6 + (prec,)
        
##########################################
##########################
        if hasattr(self.model.module, 'classifier'):
            #print("3333333333333333333333333333333333333333333333333333333333333333333333333")
            loss = self.criterion(outputs[1], targets)
            prec = accuracy(outputs[1].data, targets.data)[0]
            return (loss,)*6 + (prec,)
      #################################  
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            loss2 = self.criterion(outputs[1][2],targets)
            loss3 = self.criterion(outputs[1][3],targets)
            loss4 = self.criterion(outputs[1][4],targets)
            loss5 = self.criterion(outputs[1][5],targets)
            #prec, = accuracy(outputs[1][2].data, targets.data)
            #prec = prec[0]

            #print(f"原始损失值 - loss0: {loss0.item()}, loss1: {loss1.item()}, ...")
           # print(f"预测值范围: {outputs[1][0].min().item():.3f} ~ {outputs[1][0].max().item():.3f}")
           # print(f"标签值示例: {targets[:5].cpu().numpy()}")
            
            
            prec = accuracy(outputs[1][2].data, targets.data)[0]  
            if torch.is_tensor(prec):
                prec = prec.item()
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            #prec, = accuracy(outputs.data, targets.data)
            #prec = prec[0]
            
            prec = accuracy(outputs[1][2].data, targets.data)[0]  
            if torch.is_tensor(prec):
                prec = prec.item()

        
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, prec
