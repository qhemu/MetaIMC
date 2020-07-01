import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np
import math
from torch.optim import Adam
from collections import OrderedDict
from models import *
from train_eval import *


class Meta(nn.Module):
    """
    Meta Learner
    """
    # def __init__(self, args, config):
    def __init__(self, args, num_relations, n_features, multiply_by):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.number_of_training_steps_per_iter = args.number_of_training_steps_per_iter#10
        self.multi_step_loss_num_epochs = args.multi_step_loss_num_epochs#5
        # self.lr_decay_step_size = args.lr_decay_step_size#5
        # self.lr_decay_factor = args.lr_decay_factor#5
        # yahoo
        if args.data_name == 'yahoo_music':
            self.net = IGMC(4,   # 4 71
                     101,
                     latent_dim=[32, 32, 32, 32],
                     num_relations=num_relations,
                     num_bases=4,
                     regression=True,
                     adj_dropout=args.adj_dropout,
                     force_undirected=args.force_undirected,
                     side_features=args.use_features,
                     n_side_features=n_features,
                     multiply_by=multiply_by)
        elif args.data_name == 'douban':
            print(num_relations)
            self.net = IGMC(4,
                            6,
                            latent_dim=[32, 32, 32, 32],
                            num_relations=num_relations,
                            num_bases=4,
                            regression=True,
                            adj_dropout=args.adj_dropout,
                            force_undirected=args.force_undirected,
                            side_features=args.use_features,
                            n_side_features=n_features,
                            multiply_by=multiply_by)
        elif args.data_name == 'flixster':
            print(num_relations)
            self.net = IGMC(4,
                            10,
                            latent_dim=[32, 32, 32, 32],
                            num_relations=num_relations,
                            num_bases=4,
                            regression=True,
                            adj_dropout=args.adj_dropout,
                            force_undirected=args.force_undirected,
                            side_features=args.use_features,
                            n_side_features=n_features,
                            multiply_by=multiply_by)
        self.ARR = args.ARR
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device).reset_parameters()
        self.optimizer = Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.store_parameters()
        self.local_update_target_weight_name = ['convs.0.basis', 'convs.0.att', 'convs.0.root', 'convs.0.bias', 'convs.1.basis', 'convs.1.att', 'convs.1.root',
         'convs.1.bias', 'convs.2.basis', 'convs.2.att', 'convs.2.root', 'convs.2.bias', 'convs.3.basis', 'convs.3.att',
         'convs.3.root', 'convs.3.bias', 'lin1.weight', 'lin1.bias', 'lin2.weight', 'lin2.bias','attention.projection.0.weight', 'attention.projection.0.bias']
        #  'attention.projection.2.weight', 'attention.projection.2.bias'
        # self.local_update_target_weight_name = ['convs.0.basis', 'convs.0.att', 'convs.0.root',
        #                                     'convs.1.basis', 'convs.1.att', 'convs.1.root',
        #                                      'convs.2.basis', 'convs.2.att', 'convs.2.root',
        #                                      'convs.3.basis', 'convs.3.att',
        #                                     'convs.3.root', 'lin1.weight', 'lin2.weight',
        #                                     ]  'lin3.weight', 'lin3.bias'
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def store_parameters(self):
        self.keep_weight = deepcopy(self.net.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def get_per_step_loss_importance_vector(self,epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        生成维数张量(num_inner_loop_steps)，指示每个步骤的目标的重要性向优化方向损失。
        :返回:一个张量，用来计算损失的加权平均，对MSL(多步损耗)机制。
        """
        #"number_of_training_steps_per_iter":5,
        # "multi_step_loss_num_epochs": 10,
        #[0.2 0.2 0.2 0.2 0.2]
        loss_weights = np.ones(shape=(self.number_of_training_steps_per_iter)) * (
                1.0 / self.number_of_training_steps_per_iter)
        '''
        "number_of_training_steps_per_iter":5,
        "multi_step_loss_num_epochs": 10,
        '''
        decay_rate = 1.0 / self.number_of_training_steps_per_iter / self.multi_step_loss_num_epochs#0.02
        min_value_for_non_final_losses = 0.03 / self.number_of_training_steps_per_iter#0.03???0.006
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (epoch * (self.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        device = torch.device('cuda')
        loss_weights = torch.Tensor(loss_weights).to(device=device)
        return loss_weights

    def forward(self, epoch, x_spt, x_qry):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # x_spt = torch.from_numpy(x_spt).cuda()
        # x_qry = torch.from_numpy(x_qry).cuda()
        # aa = list(self.net.named_parameters())
        # bb = list(self.keep_weight.keys())
        total_losses=[]
        self.optimizer.zero_grad()
        self.zero_grad()
        # if (epoch+1) % self.lr_decay_step_size == 0:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.lr_decay_factor * param_group['lr']
        for i in range(self.task_num):
            losses_q = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(epoch)
            for k in range(self.update_step):
                if k > 0:
                    self.net.load_state_dict(self.fast_weights)
                weight_for_local_update = list(self.net.state_dict().values())
                loss = self.trainingout(x_spt[i], regression=True)
                self.net.zero_grad()
                grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True, create_graph=True)
                # compute fastweights -- local update
                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
                for j in range(self.weight_len):
                    if self.weight_name[j] in self.local_update_target_weight_name:
                        self.fast_weights[self.weight_name[j]] = weight_for_local_update[j] - self.update_lr * grad[j]
                    else:
                        self.fast_weights[self.weight_name[j]] = weight_for_local_update[j]
                # bb = self.fast_weights
                self.net.load_state_dict(self.fast_weights)
                # cc = list(self.net.parameters())
                loss_q = self.trainingout(x_qry[i], regression=True)
                # self.net.load_state_dict(self.keep_weight)  # old parms？
                # dd = list(self.net.parameters())

                if epoch < self.multi_step_loss_num_epochs:#10
                    losses_q.append(per_step_loss_importance_vectors[k] * loss_q)
                else:
                    #  "number_of_training_steps_per_iter":5,
                    if k == (self.number_of_training_steps_per_iter - 1):
                        losses_q.append(loss_q)#5
                # losses_q.append(loss_q)  # 5

            task_losses = torch.sum(torch.stack(losses_q))
            # total_losses.append(losses_q[-1])
            total_losses.append(task_losses)
            # del grad
            del loss_q
            del losses_q
            del task_losses

        # print(total_losses)
        # 评价指标
        # end of all tasks
        losses = torch.stack(total_losses).mean(0)
        # print(losses)
        # optimize theta parameters
        # self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.store_parameters()
        loss = losses.item()
        del losses
        del grad
        del total_losses
        del weight_for_local_update
        del per_step_loss_importance_vectors
        torch.cuda.empty_cache()
        return loss

    def finetunning(self, x_spt, x_qry):
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net = deepcopy(self.net)
        losses = list()
        self.net.eval()
        tmp = 0
        for k in range(self.update_step_test):
            if k > 0:
                self.net.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.net.state_dict().values())
            loss = self.trainingout(x_spt, regression=True)
            # loss /= torch.norm(loss).tolist()
            # loss = self.fintuneout(x_spt, regression=True)
            self.net.zero_grad()
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True, create_graph=True)
            # compute fastweights -- local update
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            for j in range(self.weight_len):
                if self.weight_name[j] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[j]] = weight_for_local_update[j] - self.update_lr * grad[j]
                else:
                    self.fast_weights[self.weight_name[j]] = weight_for_local_update[j]
            self.net.load_state_dict(self.fast_weights)
            loss_q = self.evalloss(x_qry, True, show_progress=True)
            # self.net.load_state_dict(self.keep_weight)
            losses.append(loss_q.item())
        del grad
        del loss
        del loss_q
        del weight_for_local_update
        torch.cuda.empty_cache()
        self.store_parameters()
        return min(losses)  # [-1]


    def test_once(self, test_dataset,
                  batch_size,
                  logger=None,
                  ensemble=False,
                  checkpoints=None):

        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        self.net.to(device)
        t_start = time.perf_counter()
        if ensemble and checkpoints:
            rmse = eval_rmse_ensemble(self.net, checkpoints, test_loader, device, show_progress=True)
        else:
            rmse = eval_rmse(self.net, test_loader, device, show_progress=True)
        t_end = time.perf_counter()
        duration = t_end - t_start
        print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
        epoch_info = 'test_once' if not ensemble else 'ensemble'
        eval_info = {
            'epoch': epoch_info,
            'train_loss': 0,
            'test_rmse': rmse,
        }
        if logger is not None:
            logger(eval_info, None, None)
        return rmse.item()
    def statedict(self, model_file, optimizer_name):
        torch.save(self.net.state_dict(), model_file)
        torch.save(self.optimizer.state_dict(), optimizer_name)

    def load_state_dict(self,model_file):
        self.net.load_state_dict(torch.load(model_file))
    def trainingout(self, data, regression=False):
        self.net.train()
        self.optimizer.zero_grad()
        out = self.net(data)  # shape torch.Size([50])
        if regression:
            loss = F.mse_loss(out, data.y.cuda().view(-1))
        else:
            loss = F.nll_loss(out, data.y.cuda().view(-1))
        ARR = 0.001
        if ARR != 0:  # 0.001
            for gconv in self.net.convs:
                w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations,
                                                                                        gconv.in_channels,
                                                                                        gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss += ARR * reg_loss
        '''
        model.convs:
            ModuleList(
              (0): RGCNConv(4, 32, num_relations=10)
              (1): RGCNConv(32, 32, num_relations=10)
              (2): RGCNConv(32, 32, num_relations=10)
              (3): RGCNConv(32, 32, num_relations=10)
            )
        model.lin1:256,128
        model.lin2:128,1
        '''
        # loss.backward()
        # loss = loss.item() * self.num_graphs(data)
        #     # optimizer.step()
        #     torch.cuda.empty_cache()
        # return total_loss / len(loader.dataset)
        return loss
    def fintuneout(self, data, regression=False):
        self.net.eval()
        self.optimizer.zero_grad()
        out = self.net(data)  # shape torch.Size([50])
        if regression:
            loss = F.mse_loss(out, data.y.cuda().view(-1))
        else:
            loss = F.nll_loss(out, data.y.cuda().view(-1))
        ARR = 0.001
        if ARR != 0:  # 0.001
            for gconv in self.net.convs:
                w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        return loss
    def evalloss(self, data, regression=False, show_progress=False):
        self.net.eval()
        loss = 0
        with torch.no_grad():
            out = self.net(data)
        if regression:
            loss = F.mse_loss(out, data.y.cuda().view(-1), reduction='sum').item()
        else:
            loss = F.nll_loss(out, data.y.cuda().view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
        return loss  # n×k_qry
        # return loss

    def num_graphs(self, data):
        # if data.batch is not None:
        #     return data.num_graphs
        # else:
        return data.y.size(0)

    def test_once(self, test_dataset,
                  batch_size,
                  logger=None,
                  ensemble=False,
                  checkpoints=None):
        model = self.net
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        t_start = time.perf_counter()
        if ensemble and checkpoints:
            rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
        else:
            rmse = eval_rmse(model, test_loader, device, show_progress=True)
        t_end = time.perf_counter()
        duration = t_end - t_start
        print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
        epoch_info = 'test_once' if not ensemble else 'ensemble'
        eval_info = {
            'epoch': epoch_info,
            'train_loss': 0,
            'test_rmse': rmse,
        }
        if logger is not None:
            logger(eval_info)
        return rmse

def main():
    pass


if __name__ == '__main__':
    main()
