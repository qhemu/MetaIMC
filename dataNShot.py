import torch
import  numpy as np
from util_functions import *

import random

class dataNShot:

    def __init__(self, m_trainloader, m_testloader, train_indx, test_indx, batchsz, k_shot, k_query):
        self.task_num = batchsz  # 1
        self.k_shot = k_shot  # k shot3
        self.k_query = k_query  # k query2
        metatrain = {}
        metatest = {}
        for i in range(len(train_indx)):
            metatrain.setdefault(train_indx[i], []).append(m_trainloader[i])
        # for j in range(len(test_indx)):
        #     metatest.setdefault(test_indx[j], []).append(m_testloader[j])

        print("users for train/test:", len(metatrain), "/", len(metatest))

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0}
        self.datasets = {"train": metatrain}  # original data cached
        self.datasets_cache = {
                               "train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               # "test": self.load_data_cache(self.datasets["test"])
                               }
    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        为N-shot学习收集几批数据
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :返回:一个列表，其中[support_set_x, support_set_y, target_x, target_y]准备好被提供给我们的网络
        """
        data_cache = list()
        userkeys = [i for i in data_pack.keys()]
        random.seed()
        selected_user = random.sample(userkeys, self.task_num)
        # print(selected_user)
        x_spts, x_qrys = [], []
        for i in range(self.task_num):  # one task_num means one set 10
            cur_class = selected_user[i]
            random.seed()
            selected_data = random.sample(data_pack[cur_class], self.k_shot + self.k_query)
            Metasets1 = Metaset(selected_data[:self.k_shot])
            x_spt = Metasets1.process()
            x_spts.append(x_spt)
            Metasets2 = Metaset(selected_data[self.k_shot:])
            x_qry = Metasets2.process()
            x_qrys.append(x_qry)
            # shuffle inside a batch
            # perm = np.random.permutation(self.n_way * self.k_shot)
            # x_spt = np.array(x_spt)[perm]  # 3256
            # perm = np.random.permutation(self.n_way * self.k_query)
            # x_qry = np.array(x_qry)[perm]  # 3256
            # # a batch
            # x_spts = np.array(x_spts)
            # x_qrys = np.array(x_qrys)
        data_cache.append([x_spts, x_qrys])
        del x_spts
        del x_qrys
        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        从具有名称的数据集中获取下一批数据。
        :param模式:拆分名称(其中“train”、“val”、“test”)
        """
        # update cache if indexes is larger cached num 如果索引缓存的num较大，则更新缓存
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


