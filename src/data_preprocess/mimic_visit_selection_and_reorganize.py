# No. 3
# 基于数据构建相关的标签
# 将数据的个个特征打上不同的标记
# 删除缺失值过多的访问（之所以要在这里做而非在之前做，是因为部分访问可能缺失值很多，但是其作为标签时的信息不应被直接浪费）
# 20200715复核
# 20200731复核
from util import *
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    visit_delete_missing_rate = 0.3
    figure_save_folder = os.path.abspath('../../resource/preprocessed_data/mimic_feature_distribution')
    feature_order_path = os.path.abspath('../../resource/mapping_file/mimic/feature_order.csv')
    data_path = os.path.join(os.path.abspath('../../resource/preprocessed_data/'), 'mimic_after_variable_selection.csv')

    feature_dict_list = read_feature_list(feature_order_path)
    data_dict = general_read_data_to_dict(data_path, skip_extra_line=2)
    print('data, visit: {}, variable num: {}'
          .format(calculate_visit_count(data_dict), calculate_variable_number(data_dict)))
    next_visit_dict = next_visit_index(data_dict)
    reorganized_data_dict = data_reorganization(data_dict, next_visit_dict, feature_dict_list)
    print('reorganized data, visit: {}, variable num: {}'
          .format(calculate_visit_count(reorganized_data_dict), calculate_variable_number(reorganized_data_dict)))
    reorganized_data_dict = delete_visit_missing_too_much(reorganized_data_dict, visit_delete_missing_rate)
    print('discard visit with significant missing, remaining: {}, variable num: {}'
          .format(calculate_visit_count(reorganized_data_dict), calculate_variable_number(reorganized_data_dict)))
    reorganized_data_dict = discard_visit_without_label_or_disease(reorganized_data_dict)
    print('discard visit without label or disease, remaining: {}, variable num: {}'
          .format(calculate_visit_count(reorganized_data_dict), calculate_variable_number(reorganized_data_dict)))

    save_path = os.path.join(os.path.abspath('../../resource/preprocessed_data/'),
                             'mimic_after_label_generate_and_visit_selection.csv')
    write_data_dict_to_csv(reorganized_data_dict, save_path, True, True)
    plot_all_numeric_feature_with_distribution(reorganized_data_dict, figure_save_folder)


def discard_visit_without_label_or_disease(data_dict):
    """20200731复核，删除患者中没有label的或者没有interact的"""
    discard_set = set()
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            discard_flag_interact = True
            discard_flag_label = True
            for key in data_dict[patient_id][visit_id]:
                if key.__contains__('disease') or key.__contains__('risk') or key.__contains__('category'):
                    value = float(data_dict[patient_id][visit_id][key])
                    if value > 0.5:
                        discard_flag_interact = False
                if key.__contains__('label'):
                    value = float(data_dict[patient_id][visit_id][key])
                    if value > 0.5:
                        discard_flag_label = False
            if discard_flag_interact or discard_flag_label:
                discard_set.add(patient_id+"_"+visit_id)
    for item in discard_set:
        pat_id, visit_id = item.split('_')
        data_dict[pat_id].pop(visit_id)
    print(len(discard_set))
    return data_dict


def read_feature_list(path):
    """
    由于本设计中，数据和知识图谱要搭配使用，因此特征的排列顺序要预先规定，确保idx和KG中的相应节点的idx能够严格对齐
    20200715复核
    20200731复核 确认和知识图谱（无干预版本）排序一致
    """
    feature_dict_list = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            feature_type, name = line
            if not feature_dict_list.__contains__(feature_type):
                feature_dict_list[feature_type] = list()
            feature_dict_list[feature_type].append(name)
    return feature_dict_list


def plot_all_numeric_feature_with_distribution(reorganized_data_dict, save_folder):
    # 绘制数据分布图，为接下来的分布变换做准备
    # 找出数值型变量
    numeric_feature_dict = is_feature_numerical(reorganized_data_dict)
    # 生成序列
    feature_dict = dict()
    for patient_id in reorganized_data_dict:
        for visit_id in reorganized_data_dict[patient_id]:
            for item in reorganized_data_dict[patient_id][visit_id]:
                if numeric_feature_dict[item]:
                    if not feature_dict.__contains__(item):
                        feature_dict[item] = list()
                    value = float(reorganized_data_dict[patient_id][visit_id][item])
                    if value >= 0:
                        feature_dict[item].append(value)
    # 初次值域变换
    num_dict = dict()
    for item in feature_dict:
        feature_dict[item] = np.array(sorted(feature_dict[item]))
        num_dict[item] = {'max': feature_dict[item][-1]+0.001, 'min': feature_dict[item][0]-0.001}
        feature_dict[item] = (feature_dict[item]-num_dict[item]['min'])/(num_dict[item]['max']-num_dict[item]['min'])

    for item in feature_dict:
        skip = feature_dict[item]
        log = np.log(feature_dict[item])
        arcsin = np.arcsin(feature_dict[item])**0.5
        sqrt = feature_dict[item]**0.5
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
        axs[0][0].hist(skip, bins=50)
        axs[0][1].hist(log, bins=50)
        axs[1][0].hist(sqrt, bins=50)
        axs[1][1].hist(arcsin, bins=50)
        axs[0][0].set_title('{} skip'.format(item))
        axs[0][1].set_title('{} log'.format(item))
        axs[1][0].set_title('{} sqrt'.format(item))
        axs[1][1].set_title('{} arcsin'.format(item))
        plt.savefig(os.path.join(save_folder, item+'.png'))


def delete_visit_missing_too_much(feature_dict, visit_delete_missing_rate):
    """
    20200715复核
    20200731复核
    如果一个患者的数值型数据的缺失率超过容限，则将该次数据直接删除
    """
    # 构建基本数据，及判定到底哪些特征时数值型
    feature_type_dict = dict()
    for patient_id in feature_dict:
        for visit_id in feature_dict[patient_id]:
            for item in feature_dict[patient_id][visit_id]:
                feature_type_dict[item] = False
    for patient_id in feature_dict:
        for visit_id in feature_dict[patient_id]:
            for item in feature_dict[patient_id][visit_id]:
                value = float(feature_dict[patient_id][visit_id][item])
                if value != 0 and value != 1 and value != -1:
                    feature_type_dict[item] = True
    numerical_feature_num = 0
    for item in feature_type_dict:
        if feature_type_dict[item]:
            numerical_feature_num += 1

    discard_visit_set = set()
    for patient_id in feature_dict:
        for visit_id in feature_dict[patient_id]:
            missing_count = 0
            for item in feature_dict[patient_id][visit_id]:
                if feature_type_dict[item]:
                    value = float(feature_dict[patient_id][visit_id][item])
                    if value < 0:
                        missing_count += 1
            if missing_count / numerical_feature_num > visit_delete_missing_rate:
                discard_visit_set.add(patient_id + '_' + visit_id)

    for item in discard_visit_set:
        patient_id, visit_id = item.split('_')
        feature_dict[patient_id].pop(visit_id)
    return feature_dict


def data_reorganization(data_dict, next_visit_dict, feature_dict_list):
    """
    20200715复核
    20200731复核
    """
    new_data_dict = dict()
    for patient_id in data_dict:
        new_data_dict[patient_id] = dict()
        for visit_id in data_dict[patient_id]:
            # 我们不需要没有标签的数据
            if not (next_visit_dict.__contains__(patient_id) and next_visit_dict[patient_id].__contains__(visit_id)):
                continue
            new_data_dict[patient_id][visit_id] = dict()

            # get label
            next_visit = next_visit_dict[patient_id][visit_id]
            for item in feature_dict_list['disease']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_label'] = data_dict[patient_id][next_visit][item]
            # get risk factor
            for item in feature_dict_list['risk_factor']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_risk_factor'] = data_dict[patient_id][visit_id][item]
            # get treatment
            for item in feature_dict_list['treatment']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_treatment'] = data_dict[patient_id][visit_id][item]
            # get disease
            for item in feature_dict_list['disease']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_disease'] = data_dict[patient_id][visit_id][item]
            # get disease category
            for item in feature_dict_list['category']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_category'] = data_dict[patient_id][visit_id][item]
            # get disease category
            for item in feature_dict_list['feature']:
                if data_dict[patient_id][next_visit].__contains__(item):
                    new_data_dict[patient_id][visit_id][item+'_feature'] = data_dict[patient_id][visit_id][item]
    return new_data_dict


def next_visit_index(data_dict):
    """
    20200715复核
    20200731复核
    """
    index_dict = dict()
    for patient_id in data_dict:
        index_dict[patient_id] = list()
        for visit_id in data_dict[patient_id]:
            index_dict[patient_id].append(int(visit_id))
        index_dict[patient_id] = sorted(index_dict[patient_id])

    next_visit_dict = dict()
    for patient_id in index_dict:
        if len(index_dict[patient_id]) == 1:
            continue
        next_visit_dict[patient_id] = dict()
        for index in range(len(index_dict[patient_id])-1):
            next_visit_dict[patient_id][str(index_dict[patient_id][index])] = str(index_dict[patient_id][index+1])
    return next_visit_dict


if __name__ == '__main__':
    main()
