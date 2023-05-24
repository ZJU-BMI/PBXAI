# No.4 进行分布变换与数据插补
# 20200715复核
import os
from util import general_read_data_to_dict, write_data_dict_to_csv
import csv
import math
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def main():
    file_path = os.path.abspath('../../resource/preprocessed_data/mimic_after_label_generate_and_visit_selection.csv')
    strategy_path = os.path.abspath('../../resource/mapping_file/mimic/分布变换策略.csv')
    save_un_imputed_path = os.path.abspath('../../resource/preprocessed_data/mimic_un_imputed_data.csv')
    save_imputed_path = os.path.abspath('../../resource/preprocessed_data/mimic_imputed_data.csv')
    feature_order_path = os.path.abspath('../../resource/mapping_file/mimic/feature_order.csv')

    iter_num = 100
    placeholder_replace = -99999
    data_dict = general_read_data_to_dict(file_path, skip_extra_line=2)
    feature_output_order = output_data_order_list(feature_order_path)
    print('read data')
    max_min_dict = find_max_and_minimum(data_dict)
    transform_dict = read_transform_strategy(strategy_path)
    data_dict = value_transform(data_dict, transform_dict, max_min_dict, placeholder_replace=placeholder_replace)
    write_data_dict_to_csv(data_dict, save_un_imputed_path, feature_order_list=feature_output_order)
    print('value transformed')
    reconstructed_data_dict = data_impute(data_dict, placeholder_replace=placeholder_replace, iter_num=iter_num)
    print('data imputed')
    write_data_dict_to_csv(reconstructed_data_dict, save_imputed_path, feature_order_list=feature_output_order)


def read_feature_list(path):
    """
    由于本设计中，数据和知识图谱要搭配使用，因此特征的排列顺序要预先规定，确保idx和KG中的相应节点的idx能够严格对齐
    20200715复核
    20200731复核
    20200820复核
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


def output_data_order_list(feature_order_path):
    """定义输出顺序 20200731复核"""
    order_list = list()
    feature_dict_list = read_feature_list(feature_order_path)
    for key in feature_dict_list:
        for item in feature_dict_list[key]:
            order_list.append(item+'_'+key)
            if key == 'disease':
                order_list.append(item + '_label')
    return order_list


def data_impute(data_dict, iter_num, placeholder_replace):
    patient_visit_list = list()
    feature_list = list()
    data_mat = list()

    # 填充feature_list
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for feature in data_dict[patient_id][visit_id]:
                feature_list.append(feature)
            break
        break
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            patient_visit_list.append([patient_id, visit_id])
            line = list()
            for feature in feature_list:
                line.append(float(data_dict[patient_id][visit_id][feature]))
            data_mat.append(line)

    imp = IterativeImputer(max_iter=iter_num, missing_values=placeholder_replace)
    imputed_data = imp.fit_transform(data_mat)
    reconstructed_data_dict = dict()
    for index in range(len(patient_visit_list)):
        patient_id, visit_id = patient_visit_list[index]
        if not reconstructed_data_dict.__contains__(patient_id):
            reconstructed_data_dict[patient_id] = dict()
        if not reconstructed_data_dict[patient_id].__contains__(visit_id):
            reconstructed_data_dict[patient_id][visit_id] = dict()
        for sub_idx in range(len(feature_list)):
            reconstructed_data_dict[patient_id][visit_id][feature_list[sub_idx]] = imputed_data[index][sub_idx]
    return reconstructed_data_dict


def value_transform(data_dict, transform_dict, max_min_dict, placeholder_replace):
    # 数值变换
    # 20200715复核
    # 20200731复核
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for feature in data_dict[patient_id][visit_id]:
                if not transform_dict.__contains__(feature):
                    continue
                value = float(data_dict[patient_id][visit_id][feature])
                if value < 0:
                    """由于数据值域变换后-1是可以取到的，因此我们要用一个新值做占位符"""
                    data_dict[patient_id][visit_id][feature] = placeholder_replace
                else:
                    strategy = transform_dict[feature]
                    max_num, min_num = max_min_dict[feature]['max'], max_min_dict[feature]['min']
                    value = (value-min_num)/(max_num-min_num)

                    if strategy == 'skip':
                        value = value
                    elif strategy == 'arcsin':
                        value = math.asin(value)**0.5
                    elif strategy == 'sqrt':
                        value = math.sqrt(value)
                    elif strategy == 'log':
                        value = math.log(value)
                    else:
                        raise ValueError('Error Transform Method')
                    data_dict[patient_id][visit_id][feature] = value

    # 计算转换后的均值和方差
    feature_dict = dict()
    for feature in transform_dict:
        feature_dict[feature] = list()
        for patient_id in data_dict:
            for visit_id in data_dict[patient_id]:
                value = float(data_dict[patient_id][visit_id][feature])
                if value != placeholder_replace:
                    feature_dict[feature].append(value)
    stat_dict = dict()
    for feature in feature_dict:
        value_list = np.array(feature_dict[feature])
        stat_dict[feature] = {'mean': np.mean(value_list), 'std': np.std(value_list)}

    # 转化为近似标准正态分布
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for feature in data_dict[patient_id][visit_id]:
                if not feature_dict.__contains__(feature):
                    continue
                mean, std = stat_dict[feature]['mean'], stat_dict[feature]['std']
                value = float(data_dict[patient_id][visit_id][feature])
                if value != placeholder_replace:
                    value = (value - mean) / std
                    data_dict[patient_id][visit_id][feature] = value
    return data_dict


def read_transform_strategy(path):
    """
    20200715复核
    20200731复核
    """
    transform_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            transform_dict[line[0]] = line[1]
    return transform_dict


def find_max_and_minimum(data_dict):
    """
    20200715复核
    20200731复核
    """
    num_dict = dict()
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for item in data_dict[patient_id][visit_id]:
                if not num_dict.__contains__(item):
                    num_dict[item] = list()
                value = float(data_dict[patient_id][visit_id][item])
                if value >= 0:
                    num_dict[item].append(value)
    max_min_dict = dict()
    for item in num_dict:
        num_dict[item] = sorted(num_dict[item])
        max_min_dict[item] = {'max': num_dict[item][-1]+0.001, 'min': num_dict[item][0]-0.001}
    return max_min_dict


if __name__ == '__main__':
    main()
