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
parser.add_argument('--path', type=str, default='checkpoints/checkpoint.pt')
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p') 

def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
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


def build_imagenet(model_config, model_state_dict, **kwargs):

    # build model
    logging.info('Building Model')
    model = NASNet.build_from_config(model_config)
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
    criterion = nn.CrossEntropyLoss().cuda()

    # build data loader
    logging.info('Building Data')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        validdir = os.path.join(args.data, 'valid.zip')
        if args.lazy_load:
            valid_data = utils.ZipDataset(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            valid_data = utils.InMemoryZipDataset(validdir, valid_transform, num_workers=args.num_workers)
    else:
        logging.info('Loading data from directory')
        validdir = os.path.join(args.data, 'val')
        if args.lazy_load:
            valid_data = dset.ImageFolder(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=args.num_workers)
    
    logging.info('Found %d in validation data', len(valid_data))

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    return valid_queue, model, criterion


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
    args.eval_batch_size = args.eval_batch_size * args.device_count
    
    logging.info("Args = %s", args)
    
    saved_args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
    model_config = utils.build_model_config(saved_args.arch, saved_args.width_stages, saved_args.n_cell_stages, saved_args.stride_stages, saved_args.dropout)
    valid_queue, model, criterion = build_imagenet(model_config, model_state_dict)

    valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)


if __name__ == '__main__':
    main()
