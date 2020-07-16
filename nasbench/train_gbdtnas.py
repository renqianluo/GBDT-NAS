import os
import sys
import glob
import time
import copy
import logging
import argparse
import random
import numpy as np
import utils
import lightgbm as lgb
from nasbench import api


parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--k', type=int, default=1000)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--leaves', type=int, default=31)
parser.add_argument('--num_boost_round', type=int, default=100)
args = parser.parse_args()


utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logging.info("Args = %s", args)

    nasbench = api.NASBench(os.path.join(args.data, 'nasbench_only108.tfrecord'))

    arch_pool, seq_pool, valid_accs = utils.generate_arch(args.n, nasbench, need_perf=True)
    feature_name = utils.get_feature_name()
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': args.leaves,
        'learning_rate': args.lr,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    sorted_indices = np.argsort(valid_accs)[::-1]
    arch_pool = [arch_pool[i] for i in sorted_indices]
    seq_pool = [seq_pool[i] for i in sorted_indices]
    valid_accs = [valid_accs[i] for i in sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(0)), 'w') as f:
        for arch, seq, valid_acc in zip(arch_pool, seq_pool, valid_accs):
            f.write('{}\t{}\t{}\t{}\n'.format(arch.matrix, arch.ops, seq, valid_acc))
    mean_val = 0.908192
    std_val = 0.023961
    for i in range(args.iterations):
        logging.info('Iteration {}'.format(i+1))
        normed_valid_accs = [(i-mean_val)/std_val for i in valid_accs]
        train_x = np.array(seq_pool)
        train_y = np.array(normed_valid_accs)
        
        # Train GBDT-NAS
        logging.info('Train GBDT-NAS')
        lgb_train = lgb.Dataset(train_x, train_y)

        gbm = lgb.train(params, lgb_train, feature_name=feature_name, num_boost_round=args.num_boost_round)
        gbm.save_model(os.path.join(args.output_dir, 'model.txt'))
    
        all_arch, all_seq = utils.generate_arch(None, nasbench)
        logging.info('Totally {} archs from the search space'.format(len(all_seq)))
        all_pred = gbm.predict(np.array(all_seq), num_iteration=gbm.best_iteration)
        sorted_indices = np.argsort(all_pred)[::-1]
        all_arch = [all_arch[i] for i in sorted_indices]
        all_seq = [all_seq[i] for i in sorted_indices]
        new_arch, new_seq, new_valid_acc = [], [], []
        for arch, seq in zip(all_arch, all_seq):
            if seq in seq_pool:
                continue
            new_arch.append(arch)
            new_seq.append(seq)
            new_valid_acc.append(nasbench.query(arch)['validation_accuracy'])
            if len(new_arch) >= args.k:
                break
        arch_pool += new_arch
        seq_pool += new_seq
        valid_accs += new_valid_acc
        
        sorted_indices = np.argsort(valid_accs)[::-1]
        arch_pool = [arch_pool[i] for i in sorted_indices]
        seq_pool = [seq_pool[i] for i in sorted_indices]
        valid_accs = [valid_accs[i] for i in sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i+1)), 'w') as f:
            for arch, seq, va in zip(arch_pool, seq_pool, valid_accs):
                f.write('{}\t{}\t{}\t{}\n'.format(arch.matrix, arch.ops, seq, va))

    logging.info('Finish Searching\n')    
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as f:
        for i in range(10):
            arch, seq, valid_acc = arch_pool[i], seq_pool[i], valid_accs[i]
            fs, cs = nasbench.get_metrics_from_spec(arch)
            test_acc = np.mean([cs[108][i]['final_test_accuracy'] for i in range(3)])
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(arch.matrix, arch.ops, seq, valid_acc, test_acc))
            print('{}\t{}\tvalid acc: {}\tmean test acc: {}\n'.format(arch.matrix, arch.ops, valid_acc, test_acc))
        


if __name__ == '__main__':
    main()
