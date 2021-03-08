import os
import copy
import numpy as np
import logging
import shutil
import threading
import gc
import zipfile
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import torchvision.transforms as transforms


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def move_to_cuda(obj):
    if torch.is_tensor(obj):
        if torch.cuda.is_available():
            return obj.cuda()
        return obj
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [move_to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: move_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1).mean()
        return loss
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * F.log_softmax(pred), 1))


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def shuffle_layer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[
        1] * out_h * out_w / layer.groups
    return delta_ops


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    return img.convert('RGB')


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list
        
    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            with open(path, 'rb') as f:
                                image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i+1)
                        thread = ReadImageThread(root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    

class ZipDataset(data.Dataset):
    def __init__(self, path, transform=None):
        super(ZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname):
                continue
            target = self.get_target(fname)
            item = (fname, class_to_idx[target])
            self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        with zipfile.ZipFile(self.path, 'r') as reader:
            sample = reader.read(sample)
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]
    
    @staticmethod
    def find_classes(reader):
        classes = [ZipDataset.get_target(name) for name in reader.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ReadZipImageThread(threading.Thread):
    def __init__(self, reader, fnames, class_to_idx, target_list):
        threading.Thread.__init__(self)
        self.reader = reader
        self.fnames = fnames
        self.target_list = target_list
        self.class_to_idx = class_to_idx
    
    def run(self):
        for fname in self.fnames:
            if InMemoryZipDataset.is_directory(fname):
                continue
            image = self.reader.read(fname)
            class_id = self.class_to_idx[InMemoryZipDataset.get_target(fname)]
            item = (image, class_id)
            self.target_list.append(item)


class InMemoryZipDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        reader = zipfile.ZipFile(self.path, 'r')
        classes, class_to_idx = self.find_classes(reader)
        fnames = sorted(reader.namelist())
        if num_workers == 1:
            for fname in fnames:
                if self.is_directory(fname):
                    continue
                target = self.get_target(fname)
                image = reader.read(fname)
                item = (image, class_to_idx[target])
                self.samples.append(item)
        else:
            num_files = len(fnames)
            threads = []
            res = [[] for i in range(num_workers)]
            num_per_worker = num_files // num_workers
            for i in range(num_workers):
                start_index = num_per_worker * i
                end_index = num_files if i == num_workers - 1 else (i+1) * num_per_worker
                thread = ReadZipImageThread(reader, fnames[start_index:end_index], class_to_idx, res[i])
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for item in res:
                self.samples += item
            del res, threads
            gc.collect()
        reader.close()
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]

    @staticmethod
    def find_classes(fname):
        classes = [ZipDataset.get_target(name) for name in fname.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class WrappedDataset(data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        super(WrappedDataset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = None
        self.samples = []
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        assert index in self.indices
        sample, target = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Dataset: {}\n'.format(self.dataset)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer, best_acc_top1, is_best=True):
    if hasattr(model, 'module'):
        model = model.module
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'best_acc_top1': best_acc_top1,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)
  

def load(model_path):
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None, 0
    logging.info('Found {}, loading for continue training'.format(newest_filename))
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    best_acc_top1 = state_dict.get('best_acc_top1')
    return args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1

  
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def generate_arch(n, num_layers, num_ops):
    def _get_arch():
        arch = []
        for i in range(num_layers):
            if i % 4 == 0:
                op = np.random.randint(1, num_ops)
            else:
                op = np.random.randint(1, num_ops+1)
            arch.append(op)
        return arch
    archs = []
    while len(archs) < n:
        arch = _get_arch()
        if arch not in archs:
            archs.append(arch)
    return archs


def generate_constrained_arch(n, num_layers, num_ops, pruned_operations):
    def _get_arch():
        arch = []
        for i in range(num_layers):
            candidates = []
            if i % 4 == 0:# new stage
                for j in range(num_ops-1):
                    feature = i * num_ops + j
                    if feature in pruned_operations: # operation is pruned, do not sample
                        continue
                    candidates.append(j+1)
                op = np.random.choice(candidates)
            else:
                for j in range(num_ops):
                    feature = i * num_ops + j
                    if feature in pruned_operations: # operation is pruned, do not sample
                        continue
                    candidates.append(j+1)
                op = np.random.choice(candidates)
            arch.append(op)
        return arch
    archs = []
    while len(archs) < n:
        arch = _get_arch()
        if arch not in archs:
            archs.append(arch)
    return archs
    

def convert_to_features(arch):
    res = []
    for op in arch:
        tmp = [0 for _ in range(len(OPERATIONS))]
        tmp[op-1] = 1
        res += tmp
    return res


def build_model_config(arch, width_stages, n_cell_stages, stride_stages, dropout):
    if isinstance(arch, str):
        arch = list(map(int, arch.strip().split()))
    config = OrderedDict()
    config['name'] = 'GBDTNASNets',
    config['bn'] = {
        'momentum': 0.1,
        'eps': 0.001
    }
    input_channel = 40
    first_cell_width = 24

    # first conv layer
    config['first_conv'] = OrderedDict({
        'name': 'ConvLayer',
        'kernel_size': 3,
        'stride': 2,
        'dilation': 1,
        'groups': 1,
        'bias': False,
        'has_shuffle': False,
        'in_channels': 3,
        'out_channels': input_channel,
        'use_bn': True,
        'act_func': 'relu6',
        'dropout_rate': 0,
        'ops_order': 'weight_bn_act'
    })
    
    
    # first block
    first_block = OrderedDict({
        'name': 'MobileInvertedResidualBlock',
        'mobile_inverted_conv': {
            'name': 'MBInvertedConvLayer',
            'in_channels': input_channel,
            'out_channels': first_cell_width,
            'kernel_size': 3,
            'stride': 1,
            'expand_ratio': 1
        },
        'shortcut': None
    })
    input_channel = first_cell_width

    # blocks
    config['blocks'] = []
    config['blocks'].append(first_block)

    for stage_idx, (width, n_cell, s) in enumerate(zip(width_stages, n_cell_stages, stride_stages)):
        for i in range(n_cell):
            if i == 0:
                stride = s
            else:
                stride = 1
            passed_cells = sum(n_cell_stages[:stage_idx])
            mbinvconv = OPERATIONS[arch[passed_cells + i]]
            if mbinvconv['name'] == 'ZeroLayer':
                mbinvconv_config = OrderedDict({
                    'name': mbinvconv['name'],
                    'stride': stride,
                })
            else:
                mbinvconv_config = OrderedDict({
                    'name': mbinvconv['name'],
                    'in_channels': input_channel,
                    'out_channels': width,
                    'kernel_size': mbinvconv['kernel_size'],
                    'stride': stride,
                    'expand_ratio': mbinvconv['expand_ratio'],
                })
            if stride == 1 and input_channel == width:
                shortcut_config = OrderedDict({
                    'name': 'IdentityLayer',
                    'in_channels': input_channel,
                    'out_channels': input_channel,
                    'use_bn': False,
                    'act_func': None,
                    'dropout_rate': 0,
                    'ops_order': 'weight_bn_act'
                })
            else:
                shortcut_config = None
            block_config = OrderedDict({
                'name': 'MobileInvertedResidualBlock',
                'mobile_inverted_conv': mbinvconv_config,
                'shortcut': shortcut_config,
            })
            config['blocks'].append(block_config)
            input_channel = width
    
    # feature mix layer
    last_channel = 1728
    config['feature_mix_layer'] = OrderedDict({
        'name': 'ConvLayer',
        'kernel_size': 1,
        'stride': 1,
        'dilation': 1,
        'groups': 1,
        'bias': False,
        'has_shuffle': False,
        'in_channels': input_channel,
        'out_channels': last_channel,
        'use_bn': True,
        'act_func': 'relu6',
        'dropout_rate': 0,
        'ops_order': 'weight_bn_act'
    })
    config['classifier'] = OrderedDict({
        'name': 'LinearLayer',
        'in_features': last_channel,
        'out_features': 1000,
        'bias': True,
        'use_bn': False,
        'act_func': None,
        'dropout_rate': dropout,
        'ops_order': 'weight_bn_act'
    })

    return config


OPERATIONS = {
    1: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB3 3x3', 'kernel_size':3 , 'expand_ratio':3},
    2: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB6 3x3', 'kernel_size':3 , 'expand_ratio':6},
    3: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB3 5x5', 'kernel_size':5 , 'expand_ratio':3},
    4: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB6 5x5', 'kernel_size':5 , 'expand_ratio':6},
    5: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB3 7x7', 'kernel_size':7 , 'expand_ratio':3},
    6: {'name': 'MBInvertedConvLayer', 'ref_name': 'MB6 7x7', 'kernel_size':7 , 'expand_ratio':6},
    7: {'name': 'ZeroLayer', 'ref_name': 'Zero Out'},
}
