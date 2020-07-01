import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
# import scipy.io as sio
# import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from data_utils import *
from preprocessing import *
from train_eval import *
from dataNShot import *
from meta import Meta  # maml
from models import *

parser = argparse.ArgumentParser(description='Inductive Graph-based Matrix Completion')
# general settings
parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
parser.add_argument('--no-train', action='store_true', default=False,
                    help='if set, skip the training and directly perform the \
                    transfer/ensemble/visualization')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode which uses a small number of data')
parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--data-appendix', default='', 
                    help='what to append to save-names when saving datasets')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--max-train-num', type=int, default=None, 
                    help='set maximum number of train data to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--dynamic-dataset', action='store_true', default=False,
                    help='if True, extract enclosing subgraphs on the fly instead of \
                    storing in disk; works for large datasets that cannot fit into memory')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--save-interval', type=int, default=10,
                    help='save model states every # epochs ')
# subgraph extraction settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number')
# subsample nodes per hop according to the ratio
parser.add_argument('--sample-ratio', type=float, default=1.0, 
                    help='if < 1, subsample nodes per hop according to the ratio')
parser.add_argument('--max-nodes-per-hop', default=10000, 
                    help='if > 0, upper bound the # nodes per hop by another subsampling')
parser.add_argument('--use-features', action='store_true', default=False,
                    help='whether to use node features (side information)')
# edge dropout settings0.2
parser.add_argument('--adj-dropout', type=float, default=0.4,
                    help='if not 0, random drops edges from adjacency matrix with this prob')
parser.add_argument('--force-undirected', action='store_true', default=False, 
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
# optimization settings
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")

# parser.add_argument('--lr-decay-step-size', type=int, default=50,
#                     help='decay lr by factor A every B steps')
# parser.add_argument('--lr-decay-factor', type=float, default=0.1,
#                     help='decay lr by factor A every B steps')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='batch size during training')
# 邻接额定值调整器.如果不是0，则调整与相邻评级相关的图形卷积参数W之间的差异default=0.001,
parser.add_argument('--ARR', type=float,  default=0.001,
                    help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')
# transfer learning, ensemble, and visualization settings
parser.add_argument('--transfer', default='',
                    help='if not empty, load the pretrained models in this path')
parser.add_argument('--num-relations', type=int, default=5,
                    help='if transfer, specify num_relations in the transferred model')
parser.add_argument('--multiply-by', type=int, default=1,
                    help='if transfer, specify how many times to multiply the predictions by')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='if True, load a pretrained model and do visualization exps')
parser.add_argument('--ensemble', action='store_true', default=False,
                    help='if True, load a series of model checkpoints and ensemble the results')
parser.add_argument('--standard-rating', action='store_true', default=False,
                    help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
# sparsity experiment settings
parser.add_argument('--ratio', type=float, default=1.0,
                    help="For ml datasets, if ratio < 1, downsample training data to the\
                    target ratio")

# add meta-learning
parser.add_argument('--metatest_users', type=int, default=80,)
parser.add_argument('--patience', type=int, default=15,)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=32)
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
parser.add_argument('--number_of_training_steps_per_iter', type=int, default=1)
parser.add_argument('--multi_step_loss_num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('-weight_decay', type=float, help='Weight decay (L2 loss on parameters).', default=5e-7)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=5e-5)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
parser.add_argument('--metatest_num', type=int, help='update steps for finetunning', default=30)
parser.add_argument('--metatrain_num', type=int, help='update steps for finetunning', default=150)
parser.add_argument('--use_meta', action='store_true', default=True,)
'''
    Set seeds, prepare for transfer learning (if --transfer)
'''
args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
args.data_name = 'douban'
args.epochs = 65
print(args)
random.seed(args.seed)
np.random.seed(args.seed)
args.hop = int(args.hop)  # 1
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)  # 10000

rating_map, post_rating_map = None, None
if args.standard_rating:  # fasle
    if args.data_name in ['flixster', 'ml_10m']:  # original 0.5, 1, ..., 5
        rating_map = {x: int(math.ceil(x)) for x in np.arange(0.5, 5.01, 0.5).tolist()}
    elif args.data_name == 'yahoo_music':  # original 1, 2, ..., 100
        rating_map = {x: (x-1)//20+1 for x in range(1, 101)}
    else:
        rating_map = None

if args.transfer:  # false
    if args.data_name in ['flixster', 'ml_10m']:  # original 0.5, 1, ..., 5
        post_rating_map = {x: int(i // (10 / args.num_relations)) for i, x in enumerate(np.arange(0.5, 5.01, 0.5).tolist())}
    elif args.data_name == 'yahoo_music':  # original 1, 2, ..., 100
        post_rating_map = {x: int(i // (100 / args.num_relations)) for i, x in enumerate(np.arange(1, 101).tolist())}
    else:  # assume other datasets have standard ratings 1, 2, 3, 4, 5
        post_rating_map = {x: int(i // (5 / args.num_relations)) for i, x in enumerate(np.arange(1, 6).tolist())}


'''
    Prepare train/test (testmode) or train/val/test (valmode) splits
'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
# '/home/qxxhemu/PycharmProjects/globecom2020/IGMC/results/flixster_testmode'
args.res_dir = os.path.join(args.file_dir, 'results/{}{}_{}'.format(args.data_name, args.save_appendix, val_test_appendix))
if args.transfer == '':
    args.model_pos = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(args.epochs))
else:
    # '/home/qxxhemu/PycharmProjects/globecom2020/IGMC/results/flixster_testmode/model_checkpoint40.pth'
    args.model_pos = os.path.join(args.transfer, 'model_checkpoint{}.pth'.format(args.epochs))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

if not args.keep_old and not args.transfer:
    # backup current main.py, model.py files
    copy('Main.py', args.res_dir)
    copy('util_functions.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('train_eval.py', args.res_dir)
# save command line input 保存命令行输入
# 'python /home/qxxhemu/PycharmProjects/globecom2020/IGMC/Main.py --data-name flixster --epochs 40 --testing --ensemble
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data_name == 'ml_1m' or args.data_name == 'ml_10m':
    if args.use_features:
        datasplit_path = 'raw_data/' + args.data_name + '/withfeatures_split_seed' + str(args.data_seed) + '.pickle'
    else:
        datasplit_path = 'raw_data/' + args.data_name + '/split_seed' + str(args.data_seed) + '.pickle'
elif args.use_features:  # false
    datasplit_path = 'raw_data/' + args.data_name + '/withfeatures.pickle'
else:
    # 'raw_data/flixster/nofeatures.pickle'
    datasplit_path = 'raw_data/' + args.data_name + '/nofeatures.pickle'

# flixster douban yahoo_music
if args.data_name == 'flixster' or args.data_name == 'douban' or args.data_name == 'yahoo_music':
    '''
    args.data_name:
    args.testing:
    rating_map:none
    post_rating_map:none
    '''
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti(args.data_name, args.testing, rating_map, post_rating_map, args.metatest_users)
elif args.data_name == 'ml_100k':
    print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(args.data_name, args.testing, rating_map, post_rating_map, args.ratio)
else:
    print("Using random dataset split ...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = create_trainvaltest_split(args.data_name, 1234, args.testing, datasplit_path, True, True, rating_map, post_rating_map, args.ratio)

print('All ratings are:')
print(class_values)
'''
Explanations of the above preprocessing:
    class_values are all the original continuous ratings, e.g. 0.5, 2...
    They are transformed to rating labels 0, 1, 2... acsendingly.
    Thus, to get the original rating from a rating label, apply: class_values[label]
    Note that train_labels etc. are all rating labels.
    But the numbers in adj_train are rating labels + 1, why? Because to accomodate neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
    If testing=True, adj_train will include both train and val ratings, and all train data will be the combination of train and val.
以上预处理说明：
    类别值都是原始的连续评级，例如0.5，2。。。
    它们将转换为分级标签0、1、2。。。。
    因此，要从评级标签获取原始评级，请应用：类值[标签]
    请注意，train_labels等都是评级标签。
    但是adj_train上的号码是评级标签+1，为什么？因为要容纳中立评级0！因此，要从adj_train获取任何边缘标签，请记住减1。
    如果测试=真，adj_train将包括train和val ratings，所有train数据将是train和val的组合。
'''

if args.use_features:  # False
    u_features, v_features = u_features.toarray(), v_features.toarray()
    n_features = u_features.shape[1] + v_features.shape[1]
    print('Number of user features {}, item features {}, total features {}'.format(u_features.shape[1], v_features.shape[1], n_features))
else:
    u_features, v_features = None, None
    n_features = 0

if args.debug:  # use a small number of data to debug False
    num_data = 1000
    train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
    val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
    test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]

if args.max_train_num is not None:  # sample certain number of train None
    perm = np.random.permutation(len(train_u_indices))[:args.max_train_num]
    train_u_indices = train_u_indices[torch.tensor(perm)]
    train_v_indices = train_v_indices[torch.tensor(perm)]

train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)
# print('#train: %d, #val: %d, #test: %d' % (len(train_u_indices), len(val_u_indices), len(test_u_indices)))


'''
    Extract enclosing subgraphs to build the train/test or train/val/test graph datasets.
    (Note that we must extract enclosing subgraphs for testmode and valmode separately, since the adj_train is different.)
    提取封闭子图以构建train/test或train/val/test图形数据集。
   （注意，我们必须分别提取testmode和valmode的封闭子图，因为adj_列是不同的。）
'''
train_graphs, val_graphs, test_graphs = None, None, None
'''
    data_name:'flixster'
    data_appendix:''
    val_test_appendix:'testmode'
'''
data_combo = (args.data_name, args.data_appendix, val_test_appendix)
if not args.dynamic_dataset:  # use preprocessed graph datasets (stored on disk) False
    # reprocess:False
    if args.reprocess or not os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
        # if reprocess=True, delete the previously cached data and reprocess.
        if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
            rmtree('data/{}{}/{}/train'.format(*data_combo))
        if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
            rmtree('data/{}{}/{}/val'.format(*data_combo))
        if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
            rmtree('data/{}{}/{}/test'.format(*data_combo))
        # extract enclosing subgraphs and build the datasets
        train_graphs, val_graphs, test_graphs = links2subgraphs(
                adj_train,
                train_indices, 
                val_indices, 
                test_indices,
                train_labels, 
                val_labels, 
                test_labels, 
                args.hop, 
                args.sample_ratio, 
                args.max_nodes_per_hop, 
                u_features, 
                v_features, 
                args.hop*2+1, 
                class_values, 
                args.testing)

    # if not args.testing:  # testing:True
    #     val_graphs_save = MyDataset(val_graphs, root='data/{}{}/{}/val'.format(*data_combo))
    test_graphs_save = MyDataset(test_graphs, root='data/{}{}/{}/test'.format(*data_combo))
    train_graphs_save = MyDataset(train_graphs, root='data/{}{}/{}/train'.format(*data_combo))
    # save meta_train meta_test data data_combo = (args.data_name, args.data_appendix, val_test_appendix)
    meta_train_path = 'data/{}{}/{}/train/meta_train.pkl'.format(*data_combo)
    meta_test_path = 'data/{}{}/{}/test/meta_test.pkl'.format(*data_combo)
    if os.path.exists('data/{}{}/{}/train/meta_train.pkl'.format(*data_combo)):
        with open(meta_train_path, 'rb') as ftrain:
            train_graphs = pickle.load(ftrain)
        with open(meta_test_path, 'rb') as ftest:
            test_graphs = pickle.load(ftest)
        print("load preprocess data over")
    else:
        with open(meta_train_path, 'wb') as fin:
            pickle.dump((train_graphs), fin)
        with open(meta_test_path, 'wb') as fin:
            pickle.dump((test_graphs), fin)
        print("Done!")

else:  # build dynamic datasets that extract subgraphs on the fly
    train_graphs = MyDynamicDataset(
                        'data/{}{}/{}/train'.format(*data_combo), 
                        adj_train,
                        train_indices, 
                        train_labels, 
                        args.hop, 
                        args.sample_ratio, 
                        args.max_nodes_per_hop, 
                        u_features, 
                        v_features, 
                        args.hop*2+1, 
                        class_values)
    test_graphs = MyDynamicDataset(
                        'data/{}{}/{}/test'.format(*data_combo), 
                        adj_train,
                        test_indices, 
                        test_labels, 
                        args.hop, 
                        args.sample_ratio, 
                        args.max_nodes_per_hop, 
                        u_features, 
                        v_features, 
                        args.hop*2+1, 
                        class_values)
    if not args.testing:
        val_graphs = MyDynamicDataset(
                        'data/{}{}/{}/val'.format(*data_combo), 
                        adj_train,
                        val_indices, 
                        val_labels, 
                        args.hop, 
                        args.sample_ratio, 
                        args.max_nodes_per_hop, 
                        u_features, 
                        v_features, 
                        args.hop*2+1, 
                        class_values)


print('#train: %d, #test: %d' % (len(train_graphs), len(test_graphs)))

# Determine testing data (on which data to evaluate the trained model
if not args.testing: 
    test_graphs = val_graphs

'''
    meta-training
'''

def training():
    if args.transfer:
        num_relations = args.num_relations
        multiply_by = args.multiply_by
    else:
        num_relations = len(class_values)
        multiply_by = 1
    metamodel = Meta(args, num_relations, n_features, multiply_by)
    # few-shot
    db_train = dataNShot(
        m_trainloader=train_graphs,
        m_testloader=test_graphs,
        train_indx=train_u_indices,
        test_indx=test_u_indices,
        batchsz=args.task_num,
        k_shot=args.k_spt,
        k_query=args.k_qry,
    )
    metamodel.cuda()
    t_start = time.perf_counter()
    test_loss_best = 1000
    train_patience = 0
    tra_test = list()
    for epoch in range(0, args.epochs):
        start_idx = 1
        stepnums = args.metatrain_num
        metatest_num = args.metatest_num
        # stepnums = 1
        # metatest_num = 1
        pbar = tqdm(range(start_idx, stepnums + start_idx))
        for b_idx in pbar:
            x_spt, x_qry = db_train.next('train')
            metamodel.train()
            train_loss = metamodel(epoch, x_spt, x_qry)
            train_info = {
                'epoch': epoch,
                'step': b_idx,
                'train_loss': train_loss,
            }
            pbar.set_description('Epoch {}, meta-train Step {}, mse {:.6f}'.format(*train_info.values()))
        '''verification, no gradient descent验证，微调'''
        rmse = metamodel.test_once(test_graphs_save, args.batch_size, logger)
        torch.cuda.empty_cache()
        # test_rmse = 0
        # if args.use_meta:
        #     testloss = list()
        #     pbar2 = tqdm(range(start_idx, metatest_num + start_idx))
        #     for b_idx2 in pbar2:
        #         x_spt, x_qry = db_train.next('test')
        #         for x_spt_one, x_qry_one in zip(x_spt, x_qry):
        #             losstest = metamodel.finetunning(x_spt_one, x_qry_one)
        #             testloss.append(losstest)
        #         test_info = {
        #             'epoch': epoch,
        #             'meta-test-step': b_idx2,
        #         }
        #         pbar2.set_description('Epoch {}, meta-test Step {}'.format(*test_info.values()))
        #     # [b, update_step+1]
        #     # print(testloss)
        #     testlosses = np.array(testloss).mean(axis=0).astype(np.float16)
        #     testlosses = testlosses/args.k_qry
        #     # test_rmse = math.sqrt(min(testlosses))
        #     test_rmse = math.sqrt(testlosses)
        #     eval_info = {
        #         'epoch': epoch,
        #         'train_loss': train_loss,
        #         'test_rmse': test_rmse,
        #     }
        #     logger(eval_info)
        # print('Train epoch:', epoch, "meta_Train loss:", train_loss, 'meta_Test rmse:', test_rmse, 'traditional_test rmse:', rmse, 'patience：', train_patience)
        # tra_test.append(rmse)
        '''apply early stop using val_acc_best, and save model'''
        test_loss = rmse
        if test_loss <= test_loss_best:
            test_loss_best = test_loss
            epoch_best = epoch
            train_patience = 0
            print('Saving model states...')
            model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch_best))
            optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch_best))
            metamodel.statedict(model_name, optimizer_name)
        else:
            train_patience += 1
        '''show val_acc,val_acc_best every 50 epoch'''
        if epoch == args.epochs-1:
            print("epoch_best:{}\t \t \t \t traditional-test:{}".format(epoch_best, test_loss_best))
            break
    return rmse
def logger(info):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse
            ))
if __name__ == '__main__':
#         # update learning rate and restart
    li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    '''
    args.number_of_training_steps_per_iter = 2  # Test Once RMSE: 19.865859,
    '''
    '''
        yahoo：
        # Test Once RMSE: 19.865859,
        50-Test Once RMSE: 20.140938
        15-19.901824
        16:
        4-2-1-1-10-5e-5-5e-5：19.863212025515097
        4-2-4-4-10-5e-5-5e-5：20.111730635359876
        4-2-5-5-10-5e-5-5e-5：traditional-test:19.975052965409613
        4-2-5-5-10-5e-6-5e-5：traditional-test:19.897042024095704
        4-2-5-5-10-1e-3-5e-5：20.143002404035208
        4-2-5-5-10-5e-4-5e-5：19.943495664469527
        4-2-1-1-10-5e-4-5e-5：19.821704729413216
        4-2-2-2-10-5e-4-5e-5：19.748257477915562
        4-2-3-3-10-5e-4-5e-5：19.826203537141883
        2-4-3-3-10-5e-4-5e-5：19.95895847980885
        3-3-3-3-10-5e-4-5e-5：19.960330864124565
        5-1-3-3-10-5e-4-5e-5：19.930257
        0.001
        4-2-2-2-10-5e-4-5e-5-16-10-2-2：19.824393 31
        4-2-2-10-5e-4-5e-5-15-3-3：19.888686176157353
        4-2-2-10-5e-4-5e-5-20-4-4：19.966275075131584
        4-2-2-10-5e-4-5e-5-5-1-1：19.783590446150512
        4-2-2-10-5e-4-5e-5-25-5-5：19.813536732097912

        flixster：280
        4-2-2-10-5e-4-5e-5-10-2-2：0.9002
        4-2-2-10-5e-6-5e-5-10-2-2：
        4-2-2-10-5e-5-5e-4-10-2-2：

        4-2-2-10-5e-4-5e-4-5-1-1：traditional-test:0.8912409749218629 26
        4-2-2-10-1e-3-5e-5-5-1-1：traditional-test:0.8872194753242795 38
        4-2-2-10-5e-4-5e-5-5-1-1：traditional-test:0.8900948190739918

        4-2-2-10-1e-3-5e-4-5-1-1：0.8932565190738663
        4-2-2-10-1e-3-5e-4-10-1-1：traditional-test:0.8932565190738663
        4-2-2-10-1e-4-1e-3-5-1-1：0.8883911975640728
        4-2-2-10-5e-3-5e-5-5-1-1：0.8927053142551192

        3-2-2-10-5e-4-5e-5-5-1-1：0.893527
        3-3-2-10-5e-4-5e-5-5-1-1：
        2-4-2-10-5e-4-5e-5-5-1-1：
        douban:
        280-30:
        18-2-5e-5-5e-4-10-1-1: 0.8030104317680059,
        35-5-1e-3-5e-5-10-1-1: 0.7663743591461714, 28
        35-5-5e-4-5e-5-10-1-1: 0.7680256945959573, 27
        35-5-1e-3-5e-4-10-1-1: 0.7688363578626923, 41
        30-10-5e-4-5e-5-10-1-1: 0.7912854590322335 23
        5-35-5e-4-5e-5-10-1-1:0.7847662064756328 30
        45-5-5e-4-5e-5-10-1-1:0.7830827403422073 44
        45-5-1e-3-5e-4-10-1-1:0.7830827403422073 40
        45-5-1e-3-5e-4-10-1-1:0.7851067312156917 36

        4.5 93990/13689 17887/0
        30-5-32-5e-4-5e-5: traditional-test:0.7646437311929273 32
        30-5-32-1e-3-5e-5: traditional-test:0.7621495024461499 39
        32-3-32-1e-3-5e-5: traditional-test:0.7555371812991788 43
        30-5-32-5e-4-5e-5: traditional-test:0.7615321532870275 9
        25-10-32-5e-4-5e-5: traditional-test:0.7723038276358263 38
        4.9
        32-3-16-1e-3-5e-5: 0.7580803101902325
        32-3-16-1e-3-5e-5: 0.7591977895574226
        33-2-16-1e-3-5e-5: 0.7523328673001403
        5-30-16-1e-3-5e-5: 0.7768723337557217
        10-25-16-1e-3-5e-5: 0.7893284905744206
    '''
    for i in range(5):
        if i == 0:
            pass
        elif i == 1:
            args.k_spt = 15
            args.k_qry = 20
            args.lr = 1e-3
            args.update_lr = 5e-5
        elif i == 2:
            args.k_spt = 5
            args.k_qry = 15
        elif i == 3:
            args.k_spt = 20
            args.k_qry = 15
        elif i == 4:
            args.k_spt = 30
            args.k_qry = 5
        # elif i == 5:
        #     args.k_spt = 2
        #     args.k_qry = 4


        print(args)
        training()
# if i == 0:100
#     args.number_of_training_steps_per_iter = 1  # :60	 	 	meta-test_loss_best:0.9634431255917497	 	 	 traditional-test:1.005202445186753
#     args.update_step = 1
# elif i == 1:
#     args.number_of_training_steps_per_iter = 2  # epoch_best:13	 	 	meta-test_loss_best:0.9385410886050755	 	 	 traditional-test:1.0054228661799687
#     args.update_step = 2
# elif i == 2:
#     args.number_of_training_steps_per_iter = 3  # epoch_best:27	 	 	meta-test_loss_best:0.9717694878416383	 	 	 traditional-test:1.005785587145169
#     args.update_step = 3
# if i == 0: 12-3
#     args.update_step_test = 1  # meta-0.9742785792574935  test-1.0317233762011986
# elif i == 1:
#     args.update_step_test = 2  # meta-0.9591255783264254  test-1.0559430997363466
# elif i == 2:
#     args.update_step_test = 3  # meta-0.9712668917450034  test-1.0559430997363466
# if i == 0:
#     args.lr = 5e-6  # meta-0.9857381434742191  test-0.9987185461399559
# elif i == 1:
#     args.lr = 5e-5  # meta-0.9606516343087124  test-0.9970452193558447
# elif i == 2:
#     args.lr = 5e-7  # meta-0.9790281373637838  test- 0.999171189957041
# elif i == 5:
#     args.lr = 1e-3  # meta-0.9174948909939499  test-0.9951939700207665
# '''
#     Train and apply the GNN model
# '''
# if False:
#     # DGCNN_RS GNN model
#     model = DGCNN_RS(train_graphs,
#                      latent_dim=[32, 32, 32, 1],
#                      k=0.6,
#                      num_relations=len(class_values),
#                      num_bases=4,
#                      regression=True,
#                      adj_dropout=args.adj_dropout,
#                      force_undirected=args.force_undirected)
#     # record the k used in sortpooling
#     if not args.transfer:
#         with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
#             f.write(' --k ' + str(model.k) + '\n')
#             print('k is saved.')
#




# if args.visualize:
#     model.load_state_dict(torch.load(args.model_pos))
#     visualize(model, test_graphs, args.res_dir, args.data_name, class_values, sort_by='prediction')
#     if args.transfer:
#         rmse = test_once(test_graphs, model, args.batch_size, logger)
#         print('Transfer learning rmse is: {:.6f}'.format(rmse))
# else:
#     if args.ensemble:
#         if args.data_name == 'ml_1m':
#             start_epoch, end_epoch, interval = args.epochs-15, args.epochs, 5
#         else:
#             start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10
#         if args.transfer:
#             checkpoints = [os.path.join(args.transfer, 'model_checkpoint%d.pth' %x) for x in range(start_epoch, end_epoch+1, interval)]
#             epoch_info = 'transfer {}, ensemble of range({}, {}, {})'.format(args.transfer, start_epoch, end_epoch, interval)
#         else:
#             checkpoints = [os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) for x in range(start_epoch, end_epoch+1, interval)]
#             epoch_info = 'ensemble of range({}, {}, {})'.format(start_epoch, end_epoch, interval)
#         rmse = test_once(test_graphs, model, args.batch_size, logger=None, ensemble=True, checkpoints=checkpoints)
#         print('Ensemble test rmse is: {:.6f}'.format(rmse))
#     else:
#         if args.transfer:
#             model.load_state_dict(torch.load(args.model_pos))
#             rmse = test_once(test_graphs, model, args.batch_size, logger=None)
#             epoch_info = 'transfer {}, epoch {}'.format(args.transfer, args.epoch)
#         print('Test rmse is: {:.6f}'.format(rmse))
#
#     eval_info = {
#         'epoch': epoch_info,
#         'train_loss': 0,
#         'test_rmse': rmse,
#     }
#     logger(eval_info, None, None)


'''
[380, 1465, 414, 2378, 301, 2317, 241, 679, 196, 1768, 72, 2303, 1294, 426, 1145, 709, 292, 1848, 2612, 2069, 2514, 1544, 2327, 695, 1479, 1714, 2205, 1917, 1369, 531, 1302, 1414, 466, 1252, 2938, 1504, 1746, 1611, 398, 2755, 2839, 721, 863, 1469, 82, 2805, 606, 492, 523, 2852, 965, 499, 2757, 1657, 2441, 2201, 930, 428, 2547, 895, 52, 2045, 629, 1601, 25, 1332, 185, 1989, 96, 2301, 760, 191, 268, 656, 675, 1387, 2139, 375, 1325, 173, 2251, 235, 1295, 2949, 1690, 1154, 335, 2087, 562, 821]
train ratings nums: 18474 test ratings nums: 2274 all_ratings: 20748
users for train/test: 750 / 90
Namespace(ARR=0, adj_dropout=0.2, batch_size=50, continue_from=None, data_appendix='', data_name='flixster', data_seed=1234, debug=False, dynamic_dataset=False, ensemble=True, epochs=40, force_undirected=False, hop=1, k_qry=4, k_spt=6, keep_old=False, lr=0.001, max_nodes_per_hop=10000, max_train_num=None, metatest_num=25, metatest_users=90, metatrain_num=120, multi_step_loss_num_epochs=10, multiply_by=1, no_train=False, num_relations=5, number_of_training_steps_per_iter=5, patience=15, ratio=1.0, reprocess=False, sample_ratio=1.0, save_appendix='', save_interval=10, seed=1, standard_rating=False, task_num=32, testing=True, transfer='', update_lr=0.0005, update_step=5, update_step_test=10, use_features=False, use_meta=True, visualize=False, weight_decay=5e-07)
'''
