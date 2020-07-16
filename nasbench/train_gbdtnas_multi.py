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
parser.add_argument('--output_dir', type=str, default='outputs/gbdtnas')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--k', type=int, default=1000)
parser.add_argument('--num_runs', type=int, default=500)
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

    mean_val = 0.908192
    std_val = 0.023961
    all_valid_accs = []
    all_test_accs = []
    for i in range(args.num_runs):
        logging.info('{} run'.format(i+1))
        arch_pool, seq_pool, valid_accs = utils.generate_arch(args.n, nasbench, need_perf=True)
        feature_name = utils.get_feature_name()
               
        for ii in range(args.iterations):
            normed_train_perfs = [(i-mean_val)/std_val for i in valid_accs]
            train_x = np.array(seq_pool)
            train_y = np.array(normed_train_perfs)
        
            # Train GBDT-NAS
            lgb_train = lgb.Dataset(train_x, train_y)

            gbm = lgb.train(params, lgb_train, feature_name=feature_name, num_boost_round=args.num_boost_round)

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
            valid_accs+= new_valid_acc
        
        sorted_indices = np.argsort(valid_accs)[::-1]
        best_arch = arch_pool[sorted_indices[0]]
        best_arch_valid_acc = valid_accs[sorted_indices[0]]
        fs, cs = nasbench.get_metrics_from_spec(best_arch)
        test_acc = np.mean([cs[108][i]['final_test_accuracy'] for i in range(3)])
        all_valid_accs.append(best_arch_valid_acc)
        all_test_accs.append(test_acc)
        logging.info('current valid accuracy: {}'.format(best_arch_valid_acc))
        logging.info('current mean test accuracy: {}'.format(np.mean(test_acc)))
        logging.info('average valid accuracy: {}'.format(np.mean(all_valid_accs)))
        logging.info('average mean test accuracy: {}'.format(np.mean(all_test_accs)))
        logging.info('best valid accuracy: {}'.format(np.max(all_valid_accs)))
        logging.info('best mean test accuracy: {}'.format(np.max(all_test_accs)))

    logging.info('average valid accuracy: {}'.format(np.mean(all_valid_accs)))
    logging.info('average mean test accuracy: {}'.format(np.mean(all_test_accs)))
    logging.info('best valid accuracy: {}'.format(np.max(all_valid_accs)))
    logging.info('best mean test accuracy: {}'.format(np.max(all_test_accs)))


if __name__ == '__main__':
    main()
