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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model_ws import NASNet
import lightgbm as lgb

parser = argparse.ArgumentParser(description='GBDT-NAS Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--eval_batch_size', type=int, default=512)
parser.add_argument('--layers', type=int, default=21)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])
parser.add_argument('--max_num_updates', type=int, default=20000)
parser.add_argument('--grad_clip', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--arch_pool', type=str, default=None)
parser.add_argument('--lr', type=float, default=1.6)
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--log_interval', type=int, default=100)

parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

parser.add_argument('--controller_iterations', type=int, default=3)
parser.add_argument('--controller_n', type=int, default=1000)
parser.add_argument('--controller_k', type=int, default=300)
parser.add_argument('--controller_m', type=int, default=5000)
parser.add_argument('--controller_num_boost_round', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.05)
parser.add_argument('--controller_leaves', type=int, default=31)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    valid_ratio = kwargs.pop('valid_ratio', None)
    valid_num = kwargs.pop('valid_num', None)
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
        if args.lazy_load:
            data = utils.ZipDataset(traindir)
        else:
            logging.info('Loading data into memory')
            data = utils.InMemoryZipDataset(traindir, num_workers=args.num_workers)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        if args.lazy_load:
            data = dset.ImageFolder(traindir)
        else:
            logging.info('Loading data into memory')
            data = utils.InMemoryDataset(traindir, num_workers=args.num_workers)
       
    num_data = len(data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    if valid_ratio is not None:
        split = int(np.floor(1 - valid_ratio * num_data))
        train_indices = sorted(indices[:split])
        valid_indices = sorted(indices[split:])
    else:
        assert valid_num is not None
        train_indices = sorted(indices[valid_num:])
        valid_indices = sorted(indices[:valid_num])

    train_data = utils.WrappedDataset(data, train_indices, train_transform)
    valid_data = utils.WrappedDataset(data, valid_indices, valid_transform)
    logging.info('train set = %d', len(train_data))
    logging.info('valid set = %d', len(valid_data))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=args.num_workers, drop_last=False)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
        pin_memory=True, num_workers=args.num_workers, drop_last=False)
    
    model = NASNet(args.width_stages, args.n_cell_stages, args.stride_stages, args.dropout)
    model.init_model(args.model_init)
    model.set_bn_param(0.1, 0.001)
    logging.info("param size = %d", utils.count_parameters(model))

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if args.no_decay_keys:
        keys = args.no_decay_keys.split('#')
        net_params=[model.get_parameters(keys, mode='exclude'),
                    model.get_parameters(keys, mode='include')]
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    train_criterion = utils.CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion, log_interval=100):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = utils.move_to_cuda(input)
        target = utils.move_to_cuda(target)

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits = model(input, arch)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        global_step += 1
        
        if global_step % log_interval == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', global_step, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch)))
        
        if global_step >= args.max_num_updates:
            break

    return top1.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion, log_interval=1):
    valid_acc_list = []
    #top1 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            #top1.reset()
            #for step, (input, target) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = utils.move_to_cuda(inputs)
            targets = utils.move_to_cuda(targets)
                
            logits = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)
                
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            #top1.update(prec1.item(), inputs.size(0))
            valid_acc_list.append(prec1.item()/100)
            
            logging.info('Valid %d arch %s\n loss %.2f top1 %f', i+1, ' '.join(map(str, arch)), loss, prec1.item())
        
    return valid_acc_list


def train_controller(params, train_input, train_target, num_boost_round):
    logging.info('Train data: {}'.format(len(train_input)))
    train_x = np.array(list(map(utils.convert_to_features, train_input)))
    train_y = np.array(train_target)
    lgb_train = lgb.Dataset(train_x, train_y)

    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return gbm


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    args.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.lr = args.lr
    args.batch_size = args.batch_size
    args.eval_batch_size = args.eval_batch_size
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.num_class = 1000
    args.num_ops = len(utils.OPERATIONS)

    logging.info("args = %s", args)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': args.controller_leaves,
        'learning_rate': args.controller_lr,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    
    if args.arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(lambda x:list(map(int, x.strip().split())), archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_imagenet(None, None, valid_num=5000, epoch=-1)

    
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_n, args.layers, args.num_ops)

    arch_pool = []
    arch_pool_valid_acc = []
    child_arch_pool_prob = None
    epoch = 1
    max_num_updates = args.max_num_updates

    for controller_iteration in range(args.controller_iterations+1):
        logging.info('Iteration %d', controller_iteration+1)
        num_updates = 0
        while True:
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, num_updates = child_train(train_queue, model, optimizer, num_updates, child_arch_pool+arch_pool[:200], child_arch_pool_prob, train_criterion)
            epoch += 1
            scheduler.step()
            if num_updates >= max_num_updates:
                break
    
        logging.info("Evaluate seed archs")
        arch_pool += child_arch_pool
        arch_pool_valid_acc = child_valid(valid_queue, model, arch_pool, eval_criterion)

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = list(map(lambda x:arch_pool[x], arch_pool_valid_acc_sorted_indices))
        arch_pool_valid_acc = list(map(lambda x:arch_pool_valid_acc[x], arch_pool_valid_acc_sorted_indices))
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(controller_iteration)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(controller_iteration)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if controller_iteration == args.controller_iterations:
            break
                            
        # Train GBDT
        logging.info('Train GBDT')
        inputs = arch_pool
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        targets = list(map(lambda x: (x - min_val) / (max_val - min_val), arch_pool_valid_acc))

        logging.info('Train GBDT')
        gbm = train_controller(params, inputs, targets, args.controller_num_boost_round)

        # Ranking sampled candidates
        random_arch = utils.generate_arch(args.controller_m, args.layers, args.num_ops)
        logging.info('Totally {} archs sampled from the search space'.format(len(random_arch)))
        random_arch_features = np.array(list(map(utils.convert_to_features, random_arch)))
        random_arch_pred = gbm.predict(random_arch_features, num_iteration=gbm.best_iteration)
        sorted_indices = np.argsort(random_arch_pred)[::-1]
        random_arch = [random_arch[i] for i in sorted_indices]
        new_arch = []
        for arch in random_arch:
            if arch in arch_pool:
                continue
            new_arch.append(arch)
            if len(new_arch) >= args.controller_k:
                break
        #arch_pool += new_arch

        logging.info("Generate %d new archs", len(new_arch))
        child_arch_pool = new_arch

    logging.info('Finish Searching')
  

if __name__ == '__main__':
    main()
