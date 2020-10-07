import csv
import numpy as np
from itertools import islice


def load_file_purely_data(path, skip_line, skip_column):
    data = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, skip_line, None):
            data_line = []
            for idx in range(skip_column, len(line)):
                data_line.append(line[idx])
            data.append(data_line)
    return np.array(data)


def save_file(pat_visit_list, feature_list, index_name_dict, save_name):
    """
    20200715复核
    20200731复核
    """
    data_to_write = []
    head = ['patient_id', 'visit_id']
    index_list = sorted(list(index_name_dict.keys()))
    for idx in index_list:
        head.append(index_name_dict[idx])
    data_to_write.append(head)
    for idx in range(len(pat_visit_list)):
        line = list()
        line.append(pat_visit_list[idx][0])
        line.append(pat_visit_list[idx][1])
        for item in feature_list[idx]:
            line.append(item)
        data_to_write.append(line)
    with open(save_name, 'w', encoding='utf-8-sig', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def get_index_name_dict(path, skip_feature=2):
    """
    读取第一行的列名，skip_feature=2代表前两列一般是pat_id和v_id需要跳过
    :return:
    """
    index_name_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            for index in range(skip_feature, len(line)):
                index_name_dict[index] = line[index]
            break
    return index_name_dict


def get_name_index_dict(path, skip_feature=2):
    name_index_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            for index in range(skip_feature, len(line)):
                name_index_dict[line[index]] = index
            break
    return name_index_dict


def write_data_dict_to_csv(data_dict, file_path, missing_rate=False, median=False, feature_order_list=None):
    """
    20200715复核
    20200731复核
    20200820复核
    """
    if feature_order_list is None:
        general_list = list()
        for patient_id in data_dict:
            for visit_id in data_dict[patient_id]:
                for item in data_dict[patient_id][visit_id]:
                    general_list.append(item)
                break
            break
    else:
        general_list = feature_order_list

    # get head
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in general_list:
        head.append(item)
    data_to_write.append(head)

    if missing_rate:
        # 计算各个指标的缺失率
        missing_dict = dict()
        general_visit_count = 0
        for item in general_list:
            missing_dict[item] = 0
        for patient_id in data_dict:
            for visit_id in data_dict[patient_id]:
                general_visit_count += 1
                for item in data_dict[patient_id][visit_id]:
                    value = data_dict[patient_id][visit_id][item]
                    if str(value).__contains__("-1"):
                        missing_dict[item] += 1
        missing_line = ['missing rate', '']
        for item in general_list:
            missing_line.append(missing_dict[item]/general_visit_count)
        data_to_write.append(missing_line)

    if median:
        # 计算各个指标的中位数
        median_dict = dict()
        for item in general_list:
            median_dict[item] = list()
        for patient_id in data_dict:
            for visit_id in data_dict[patient_id]:
                for item in data_dict[patient_id][visit_id]:
                    value = data_dict[patient_id][visit_id][item]
                    if not str(value).__contains__("-1"):
                        median_dict[item].append(value)
        for item in median_dict:
            median_dict[item] = sorted(median_dict[item])
        median = ['median', '']
        for item in general_list:
            median.append(median_dict[item][len(median_dict[item])//2])
        data_to_write.append(median)

    for patient_id in data_dict:
        visit_list = sorted([int(key) for key in data_dict[patient_id]])
        # 强制有序输出
        for i in visit_list:
            visit_id = str(i)
            row = [patient_id, visit_id]
            for key in general_list:
                row.append(data_dict[patient_id][visit_id][key])
            data_to_write.append(row)
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def general_read_data_to_dict(file_path, skip_extra_line=0):
    """
    提供通用的将数据读取为字典形式的格式
    要求：第一行必须是Feature
    前两列分别为Patient_ID，Visit_ID
    Feature行到数据行间，可以允许若干行的统计描述（也就是所谓的extra_line）
    20200715 正确性复核
    20200731 复核
    20200820 复核
    """
    data_dict = dict()
    index_name_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        line_index = 0
        for line in csv_reader:
            if line_index == 0:
                line_index += 1
                for index in range(2, len(line)):
                    index_name_dict[index] = line[index]
                continue
            if line_index <= skip_extra_line:
                line_index += 1
                continue
            patient_id, visit_id = line[0], line[1]
            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            if not data_dict[patient_id].__contains__(visit_id):
                data_dict[patient_id][visit_id] = dict()
            for index in range(2, len(line)):
                data_dict[patient_id][visit_id][index_name_dict[index]] = line[index]
    return data_dict


def calculate_visit_count(visit_dict):
    count = 0
    for patient_id in visit_dict:
        for _ in visit_dict[patient_id]:
            count += 1
    return count


def calculate_variable_number(visit_dict):
    count = 0
    for patient_id in visit_dict:
        for visit_id in visit_dict[patient_id]:
            for _ in visit_dict[patient_id][visit_id]:
                count += 1
            break
        break
    return count


def is_feature_numerical(reorganized_data_dict):
    """
    20200715复核
    20200731复核
    """
    numeric_feature_dict = dict()
    for patient_id in reorganized_data_dict:
        for visit_id in reorganized_data_dict[patient_id]:
            for item in reorganized_data_dict[patient_id][visit_id]:
                if not numeric_feature_dict.__contains__(item):
                    numeric_feature_dict[item] = False
                try:
                    value = float(reorganized_data_dict[patient_id][visit_id][item])
                    if value != 0 and value != 1 and value != -1:
                        numeric_feature_dict[item] = True
                except ValueError:
                    continue

    return numeric_feature_dict
