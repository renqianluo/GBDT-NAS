import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNet

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data', type=str, default='data/imagenet')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False) # best practice: do not use lazy_load. when using zip_file, do not use lazy_load
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias', 'bn#classifier'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=16)

parser.add_argument('--width_stages', type=str, default='32,56,112,128,256,432')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(train_queue, model, optimizer, scheduler, global_step, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = utils.move_to_cuda(input)
        target = utils.move_to_cuda(target)
    
        #optimizer.zero_grad()
        model.zero_grad()
        logits = model(input)
        global_step += 1
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
    
        if (step+1) % 100 == 0:
            lr = scheduler.get_lr()[0]
            logging.info('train %03d lr %e loss %e top1 %f top5 %f', step+1, lr, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = utils.move_to_cuda(input)
            target = utils.move_to_cuda(target)
        
            logits = model(input)
            loss = criterion(logits, target)
        
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
            if (step+1) % 100 == 0:
                logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def build_imagenet(model_config, model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')
    step = kwargs.pop('step')

    # build model
    logging.info('Building Model')
    model = NASNet.build_from_config(model_config)
    model.init_model(args.model_init)
    model.set_bn_param(model_config['bn']['momentum'], model_config['bn']['eps'])
    print(model.config)
    logging.info("param size = %d", utils.count_parameters(model))
    logging.info("multi adds = %fM", model.get_flops(torch.ones(1, 3, 224, 224).float())[0] / 1000000)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    # build criterion
    logging.info('Building Criterion')
    train_criterion = utils.CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    # build optimizer
    logging.info('Building Optimizer')
    if args.no_decay_keys:
        keys = args.no_decay_keys.split('#')
        net_params=[model.module.get_parameters(keys, mode='exclude'),
                    model.module.get_parameters(keys, mode='include')]
        optimizer = torch.optim.SGD([
            {'params': net_params[0], 'weight_decay': args.weight_decay},
            {'params': net_params[1], 'weight_decay': 0},],
            args.lr,
            momentum=0.9,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # build data loader
    logging.info('Building Data')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        validdir = os.path.join(args.data, 'valid.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
            valid_data = utils.ZipDataset(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=args.num_workers)
            valid_data = utils.InMemoryZipDataset(validdir, valid_transform, num_workers=args.num_workers)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'val')
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
            valid_data = dset.ImageFolder(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=args.num_workers)
            valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=args.num_workers)
    
    logging.info('Found %d in training data', len(train_data))
    logging.info('Found %d in validation data', len(valid_data))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    # build lr scheduler
    logging.info('Building LR Scheduler')
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs)*len(train_queue), 0, step)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, args.gamma, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    
    args.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.lr = args.lr
    args.batch_size = args.batch_size
    args.eval_batch_size = args.eval_batch_size
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    
    logging.info("Args = %s", args)
    
    old_args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
    model_config = utils.build_model_config(args.arch, args.width_stages, args.n_cell_stages, args.stride_stages, args.dropout)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_imagenet(model_config, model_state_dict, optimizer_state_dict, epoch=epoch-1, step=step-1)

    while epoch < args.epochs:
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj, step = train(train_queue, model, optimizer, scheduler, step, train_criterion)
        logging.info('train_acc %f', train_acc)
        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save(args.output_dir, args, model, epoch, step, optimizer, best_acc_top1, is_best)
        

if __name__ == '__main__':
    main()
