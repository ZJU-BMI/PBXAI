import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, coverage_error, label_ranking_loss, \
    hamming_loss
import sys
import csv
import logging
import logging.handlers
from itertools import islice
import random
"""20200715复核"""

RELATION = 'relation'
DISEASE = 'disease'
DISEASE_CATEGORY = 'disease_category'
RISK_FACTOR = 'risk_factor'
GENERAL_CONCEPT = 'general_concept'
SUB_CONCEPT = 'sub_concept'
CAUSE = 'cause'
SELF_LOOP = 'self_loop'
PATIENT = 'patient'
HAVE = 'have'

entity_type_list = [DISEASE_CATEGORY, DISEASE, RISK_FACTOR]
relation_list = [GENERAL_CONCEPT, SUB_CONCEPT, CAUSE]
kg_relation = {
    DISEASE_CATEGORY: {
        SUB_CONCEPT: [DISEASE, DISEASE_CATEGORY],
        GENERAL_CONCEPT: [DISEASE_CATEGORY],
        CAUSE: [DISEASE, DISEASE_CATEGORY]
    },
    DISEASE: {
        GENERAL_CONCEPT: [DISEASE_CATEGORY],
        CAUSE: [DISEASE, DISEASE_CATEGORY]
    },
    RISK_FACTOR: {
        CAUSE: [DISEASE, DISEASE_CATEGORY]
    }
}

relation_embed = {
    GENERAL_CONCEPT: np.array([0, 0, 0, 1]),
    SUB_CONCEPT: np.array([0, 0, 1, 0]),
    CAUSE: np.array([0, 1, 0, 0]),
    HAVE: np.array([1, 0, 0, 0]),
    SELF_LOOP: np.array([0, 0, 0, 0]),
}

reward_dict = {}
with open('../../resource/reward_set.csv', 'r', encoding='utf-8-sig') as file:
    csv_reader = csv.reader(file)
    for idx, line in enumerate(islice(csv_reader, 1, None)):
        reward_dict[idx] = {'reverse': float(line[2]), '0.5': float(line[3]), '0.75': float(line[4])}


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def integrate_data(save_folder, *file_name_list):
    data_dict = dict()
    for file_name in file_name_list:
        data_dict[file_name] = list()
        for i in range(5):
            data = np.array(np.load(os.path.join(save_folder, file_name + '_{}.npy'.format(i))), dtype=np.float)
            data_dict[file_name].append(data)
    return data_dict


class FiveFoldValidationDataPrepare(object):
    """
    感觉重复多轮五折交叉验证也没啥意思，我们还是就做一轮五折交叉验证
    20200715复核
    """
    def __init__(self, save_folder_1, save_folder_2, dataset, data_fraction, threshold=5, omit_duplicate=True):
        self._label = []
        self._feature = []
        self._data_fraction = data_fraction

        exclude_label_idx = set()

        for i in range(5):
            label = np.array(np.load(os.path.join(save_folder_1, 'label_list_{}.npy'.format(i))), dtype=np.int)
            if omit_duplicate:
                disease = np.array(np.load(os.path.join(save_folder_1, 'disease_list_{}.npy'.format(i))), dtype=np.int)
                label = label - label * disease
            for j in range(len(label[0])):
                if label[:, j].sum() < threshold:
                    exclude_label_idx.add(j)

        for i in range(5):
            label = np.array(np.load(os.path.join(save_folder_1, 'label_list_{}.npy'.format(i))), dtype=np.int)
            disease = np.array(np.load(os.path.join(save_folder_1, 'disease_list_{}.npy'.format(i))), dtype=np.int)
            label_list = list()
            for j in range(len(label)):
                single_line_1 = list()
                single_line_2 = list()
                for k in range(len(label[0])):
                    if k not in exclude_label_idx:
                        single_line_1.append(label[j][k])
                        single_line_2.append(disease[j][k])
                if omit_duplicate:
                    line_1, line_2 = np.array(single_line_1), np.array(single_line_2)
                    label_list.append(line_1 - line_1*line_2)
                else:
                    label_list.append(single_line_1)

            label_list = np.array(label_list)
            self._label.append(label_list)
            # _pat_representation_ _pat_repre_raw_
            feature = np.array(np.load(os.path.join(save_folder_2, '{}_pat_representation_{}.npy'
                                                    .format(dataset, i))))[:len(label_list)]
            self._feature.append(feature)

        count_label = 0
        count_patient = 0
        for item in self._label:
            count_patient += len(item)
            count_label += np.sum(item)
        print(count_label/count_patient)

        # 建立group idx
        group_idx_map = {'all': [], 'group_0': [], 'group_1': [], 'group_2': []}
        idx_list = np.sum(np.concatenate([item for item in self._label], axis=0), axis=0)
        list_tuple = list()
        for i, item in enumerate(idx_list):
            list_tuple.append((i, item))
        list_tuple_sorted = sorted(list_tuple, key=lambda x: x[1], reverse=True)
        for i, item in enumerate(list_tuple_sorted):
            group_idx_map['all'].append(item[0])
            if i < len(list_tuple_sorted) / 3:
                group_idx_map['group_0'].append(item[0])
            elif i < 2 * len(list_tuple_sorted) / 3:
                group_idx_map['group_1'].append(item[0])
            else:
                group_idx_map['group_2'].append(item[0])
        self.group = group_idx_map

    def get_data(self, test_index):
        idx_l = []
        for i in range(5):
            if i != test_index:
                idx_l.append(i)

        train_f = self._feature[idx_l[0]], self._feature[idx_l[1]], self._feature[idx_l[2]], self._feature[idx_l[3]]
        train_l = self._label[idx_l[0]], self._label[idx_l[1]], self._label[idx_l[2]], self._label[idx_l[3]]
        test_f = self._feature[test_index]
        test_l = self._label[test_index]
        train_f = np.concatenate(train_f, axis=0)
        train_l = np.concatenate(train_l, axis=0)
        index = [i for i in range(len(train_f))]
        random.shuffle(index)
        index = index[: int(len(train_f)*self._data_fraction)]
        train_f = train_f[index]
        train_l = train_l[index]
        return train_f, train_l, test_f, test_l


def metric(pred_prob, label, inclusion_index_set, threshold=0.5):
    # label, pred_prob structure: [n_classes, n_samples]
    included_pred_prob = list()
    included_label = list()
    for index in inclusion_index_set:
        included_pred_prob.append(pred_prob[index])
        included_label.append(label[index])
    prob = np.array(included_pred_prob).transpose()
    pred = np.array(included_pred_prob).transpose() > threshold
    true = np.array(included_label).transpose()

    micro_auc = roc_auc_score(true, prob, average='micro')
    macro_auc = roc_auc_score(true, prob, average='macro')
    micro_f1 = f1_score(true, pred, average='micro')
    macro_f1 = f1_score(true, pred, average='macro')
    micro_avg_precision = average_precision_score(true, prob, average='micro')
    macro_avg_precision = average_precision_score(true, prob, average='macro')
    coverage = coverage_error(true, prob)
    ranking_loss = label_ranking_loss(true, prob)
    hamming = hamming_loss(true, pred)
    fuse = np.concatenate([prob[:, :, np.newaxis], true[:, :, np.newaxis]], axis=2).transpose([1, 0, 2])
    top_1_num = top_k_num(fuse, 1)
    top_3_num = top_k_num(fuse, 3)
    top_5_num = top_k_num(fuse, 5)
    top_10_num = top_k_num(fuse, 10)
    top_20_num = top_k_num(fuse, 20)
    top_30_num = top_k_num(fuse, 30)
    top_40_num = top_k_num(fuse, 40)
    top_50_num = top_k_num(fuse, 50)

    return macro_auc, micro_auc, micro_f1, macro_f1, micro_avg_precision, macro_avg_precision, coverage, ranking_loss, \
        hamming, top_1_num, top_3_num, top_5_num, top_10_num, top_20_num, top_30_num, top_40_num, top_50_num


def top_k_num(fuse_result, top_num):
    hit_sum = 0
    for i in range(len(fuse_result[0])):
        result = fuse_result[:, i, :]
        result = result[result[:, 0].argsort()]
        hit_sum += np.sum(result[-1*top_num:, 1])
    return hit_sum/len(fuse_result[0])


def index_divide(label, threshold=5):
    # label [fold_idx, n_samples, n_classes]
    # 对于某些数量过少的标签，进行剔除，然后将其分组(每组都必须超过限值)
    # 从而做出四组标签：全部，Top 1/3,中部1/3,底部1/3
    # 用于评估模型的区分能力和对罕见病的判断能力
    # 20200715复核
    # 20200822复核
    exclude_label_idx = set()
    for i in range(5):
        for j in range(len(label[i][0])):
            occur = label[i][:, j]
            if occur.sum() < threshold:
                exclude_label_idx.add(j)

    # 建立group idx
    group_idx_map = {'all': [], 'group_0': [], 'group_1': [], 'group_2': []}
    idx_list = np.sum(np.concatenate([item for item in label], axis=0), axis=0)
    list_tuple = list()
    for i, item in enumerate(idx_list):
        if i not in exclude_label_idx:
            list_tuple.append((i, item))
    list_tuple_sorted = sorted(list_tuple, key=lambda x: x[1], reverse=True)
    for i, item in enumerate(list_tuple_sorted):
        group_idx_map['all'].append(item[0])
        if i < len(list_tuple_sorted) / 3:
            group_idx_map['group_0'].append(item[0])
        elif i < 2 * len(list_tuple_sorted) / 3:
            group_idx_map['group_1'].append(item[0])
        else:
            group_idx_map['group_2'].append(item[0])

    return group_idx_map


def load_kg(path):
    with open(path, 'rb') as f:
        kg = pickle.load(f)
    return kg


def load_embed(path):
    embed = np.load(path)
    return embed


def main():
    dataset = 'plagh'
    folder = os.path.abspath('../resource/preprocessed_data/{}_two_part_five_fold'.format(dataset))
    data_source = FiveFoldValidationDataPrepare(folder, 'feature_list', 'label_list')
    group = index_divide(integrate_data(folder, 'feature_list', 'label_list')['label_list'])
    for i in range(5):
        fuse_train_data_dict, test_data_dict = data_source.get_data(i)
        label = test_data_dict['label_list'].transpose()
        pred_test = np.random.random(label.shape)
        pred_test = (pred_test-np.min(pred_test, axis=0))/(np.max(pred_test, axis=0)-np.min(pred_test, axis=0))

        for key in group:
            print(metric(pred_test, label, group[key]))


if __name__ == '__main__':
    main()
