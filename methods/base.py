import os 
import time 
import torch.nn as nn 
import torch.optim as optim
import torch

from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import logging

from networks.resnet_org import model_dict 
from networks.resnet_cifar import model_dict as model_dict_cifar
from utils.util import AverageMeter, save_model
from utils.knn import knn_monitor
from tqdm import tqdm
import torch.nn.functional as F



def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class CLModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.method = args.method
        self.arch = args.arch 
        self.dataset = args.dataset

        if 'cifar' in self.dataset or 'gtsrb' in self.dataset:
            print('CIFAR-variant Resnet is loaded')
            model_fun, feat_dim= model_dict_cifar[self.arch]
            self.mlp_layers = 2
        else:
            print('Original Resnet is loaded')
            model_fun, feat_dim = model_dict[self.arch]
            self.mlp_layers = 3
        
        self.model_generator = model_fun
        self.backbone = model_fun()
        #self.distill_backbone = model_fun()
        self.feat_dim = feat_dim    
        
    def forward(self, x):
        pass 
    
    def loss(self, reps):
        pass



class CLTrainer():
    def __init__(self, args):
        self.args = args 
        #self.tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)
        self.tb_logger = SummaryWriter(log_dir=args.saved_path)
        logging.basicConfig(filename=os.path.join(self.tb_logger.log_dir, 'training.log'), level=logging.DEBUG)
        logging.info(str(args))

        self.args.warmup_epoch = 10

    def train(self, model, optimizer, train_loader,  test_loader, memory_loader, train_sampler, train_transform):
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.warmup_epoch,
                                                  after_scheduler=cosine_scheduler)



        knn_acc = 0.
        for epoch in range(self.args.start_epoch, self.args.epochs):
            model.train()

            losses = AverageMeter()
            cl_losses = AverageMeter()

            if self.args.distributed:
                train_sampler.set_epoch(epoch)

            optimizer.zero_grad()
            optimizer.step()
            warmup_scheduler.step(epoch)
            train_transform = train_transform.cuda(self.args.gpu)


            # 1 epoch training
            start = time.time()

            for i, (images, _, _) in enumerate(train_loader):

                images = images.cuda(self.args.gpu, non_blocking=True)





                v1 = train_transform(images)
                v2 = train_transform(images)


                # compute representations
                if self.args.method == 'simclr':
                    features = model(v1, v2)

                    loss, _, _ = model.criterion(features)

                elif self.args.method == 'simsiam':
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == 'byol':
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == 'moco':
                    pass
                # loss = model(v1, v2)

                # loss = model.loss(reps)

                losses.update(loss.item(), images[0].size(0))
                cl_losses.update(loss.item(), images[0].size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            # KNN-eval
            if self.args.knn_eval_freq != 0 and epoch % self.args.knn_eval_freq == 0:
                knn_acc = knn_monitor(model.backbone, memory_loader, test_loader, epoch, classes=self.args.num_classes,
                                      subset= False)

            print('[{}-epoch] time:{:.3f} | knn acc: {:.3f} | loss:{:.3f} | cl_loss:{:.3f}'.format(epoch + 1,
                                                                                                   time.time() - start,
                                                                                                   knn_acc, losses.avg,
                                                                                                   cl_losses.avg))



            # Save
            if not self.args.distributed or (self.args.distributed
                                                             and self.args.rank % self.args.ngpus_per_node == 0):
                # save model
                if (epoch + 1) % self.args.save_freq == 0:
                    save_model({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, filename=os.path.join(self.args.saved_path, 'epoch_%s.pth.tar' % (epoch + 1)))

                    print('{}-th epoch saved'.format(epoch + 1))
                # save log
                self.tb_logger.add_scalar('train/total_loss', losses.avg, epoch)
                self.tb_logger.add_scalar('train/cl_loss', cl_losses.avg, epoch)
                self.tb_logger.add_scalar('train/knn_acc', knn_acc, epoch)

                self.tb_logger.add_scalar('lr/cnn', optimizer.param_groups[0]['lr'], epoch)

        # Save final model
        if not self.args.distributed or (self.args.distributed
                                                         and self.args.rank % self.args.ngpus_per_node == 0):
            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(self.args.saved_path, 'last.pth.tar'))

            print('{}-th epoch saved'.format(epoch + 1))



    def train_freq(self, model, optimizer,   train_transform,  poison):



        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.warmup_epoch,
                                                  after_scheduler=cosine_scheduler)



        train_loader = poison.train_pos_loader
        test_loader = poison.test_loader
        test_back_loader = poison.test_pos_loader

        knn_acc = 0.


        iter_num = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            losses = AverageMeter()
            cl_losses = AverageMeter()



            cl_losses_poison = AverageMeter()
            cl_losses_clean_train =AverageMeter()


            train_transform = train_transform.cuda(self.args.gpu)

            # 1 epoch training
            start = time.time()


            for i, (images, __, _) in enumerate(train_loader):  #frequency backdoor has been injected
                #print(i)
                model.train()
                images = images.cuda(self.args.gpu, non_blocking=True)

                #data
                v1 = train_transform(images)
                v2 = train_transform(images)


                if self.args.method == 'simclr':
                    features = model(v1, v2)

                    loss, _, _ = model.criterion(features)

                elif self.args.method == 'simsiam':
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == 'byol':
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == 'moco':

                    loss = model(v1, v2)






                # loss = model(v1, v2)

                # loss = model.loss(reps)
                losses.update(loss.item(), images[0].size(0))
                cl_losses.update(loss.item(), images[0].size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()




            warmup_scheduler.step()
            # KNN-eval
            # epoc
            if self.args.poison_knn_eval_freq != 0 and epoch % self.args.poison_knn_eval_freq == 0:
                knn_acc, back_acc = self.knn_monitor_fre(model.module.backbone if self.args.distributed else model.backbone, poison.memory_loader, test_loader, epoch, self.args,
                                                     classes=self.args.num_classes,
                                                     subset=False,
                                                     backdoor_loader=test_back_loader,
                                                     )

            print('[{}-epoch] time:{:.3f} | knn acc: {:.3f} | back acc: {:.3f} | loss:{:.3f} | cl_loss:{:.3f}'.format(
                epoch + 1,
                time.time() - start,
                knn_acc, back_acc, losses.avg,
                cl_losses.avg))


            start1 = time.time()


            # Save
            if not self.args.distributed or (self.args.distributed
                                                             and self.args.local_rank % self.args.ngpus_per_node == 0):
                # save model
                start2 = time.time()
                if epoch % self.args.save_freq == 0:
                    save_model({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, filename=os.path.join(self.args.saved_path, 'epoch_%s.pth.tar' % (epoch + 1)))

                    print('{}-th epoch saved'.format(epoch + 1))
                # save log
                self.tb_logger.add_scalar('train/total_loss', losses.avg, epoch)
                self.tb_logger.add_scalar('train/cl_loss', cl_losses.avg, epoch)
                self.tb_logger.add_scalar('train/knn_acc', knn_acc, epoch)
                self.tb_logger.add_scalar('train/back_acc', back_acc, epoch)
                self.tb_logger.add_scalar('lr/cnn', optimizer.param_groups[0]['lr'], epoch)



        # Save final model
        if not self.args.distributed or (self.args.distributed
                                                         and self.args.local_rank % self.args.ngpus_per_node == 0):
            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(self.args.saved_path, 'last.pth.tar'))

            print('last epoch saved')

    @torch.no_grad()
    def knn_monitor_fre(self, net, memory_data_loader, test_data_loader, epoch, args, k=200, t=0.1, hide_progress=True,
                         classes=-1, subset=False, backdoor_loader=None):

        net.eval()

        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        # generate feature bank
        for data, target, _ in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            feature = net(data.cuda(non_blocking=True))

            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # feature_bank: [dim, total num]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # feature_labels: [total num]

        feature_labels =  torch.tensor(memory_data_loader.dataset[:][1], device=feature_bank.device)


        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target, _ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            # feature: [bsz, dim]
            pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})

        # frequency test data

            # if args.threatmodel == 'single-class' or args.threatmodel == 'single-poison':
        if backdoor_loader is not None:

            backdoor_top1, backdoor_num = 0.0, 0
            backdoor_test_bar = tqdm(backdoor_loader, desc='kNN', disable=hide_progress)
            for data, target, _ in backdoor_test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                feature = net(data)
                feature = F.normalize(feature, dim=1)
                # feature: [bsz, dim]
                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, k, t)

                backdoor_num += data.size(0)
                backdoor_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_postfix({'Accuracy': backdoor_top1 / backdoor_num * 100})


            return total_top1 / total_num * 100, backdoor_top1 / backdoor_num * 100

        return total_top1 / total_num * 100


    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # feature: [bsz, dim]
        # feature_bank: [dim, total_num]
        # feature_labels: [total_num]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # sim_matrix: [bsz, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

        # sim_labels: [bsz, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # one_hot_label: [bsz*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [bsz, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

