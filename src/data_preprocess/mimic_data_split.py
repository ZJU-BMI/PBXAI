from util import *
import random
import os
# 20200715复核
# 20200731复核


def read_data_split_to_feature_and_label(path):
    pat_visit_list = list()
    feature_list = list()
    label_list = list()

    feature_index_name_dict = dict()
    label_index_name_dict = dict()
    general_index_category_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        feature_idx = 0
        label_idx = 0
        line_idx = 0
        for line in csv_reader:
            if line_idx == 0:
                line_idx += 1
                # 跳过前两个是因为前两个是p_id和v_id
                for idx in range(2, len(line)):
                    name = line[idx]
                    if name.__contains__('label'):
                        general_index_category_dict[idx] = 'label'
                        label_index_name_dict[label_idx] = name
                        label_idx += 1
                    elif name.__contains__('feature') or name.__contains__('risk'):
                        general_index_category_dict[idx] = 'feature'
                        feature_index_name_dict[feature_idx] = name
                        feature_idx += 1
                continue
            pat_visit_list.append([line[0], line[1]])
            label_line = list()
            feature_line = list()
            for idx in range(2, len(line)):
                if not general_index_category_dict.__contains__(idx):
                    continue
                if general_index_category_dict[idx] == 'label':
                    label_line.append(line[idx])
                elif general_index_category_dict[idx] == 'feature' or general_index_category_dict[idx] == 'risk'\
                        or general_index_category_dict[idx] == 'category' or \
                        general_index_category_dict[idx] == 'disease':
                    feature_line.append(line[idx])
            feature_list.append(feature_line)
            label_list.append(label_line)
    return feature_list, label_list, pat_visit_list, label_index_name_dict, feature_index_name_dict


def read_data_split_to_feature_risk_treatment_label(path):
    """
    20200715复核
    20200731复核
    """
    pat_visit_list = list()
    feature_list = list()
    risk_factor_list = list()
    label_list = list()
    treatment_list = list()
    disease_list = list()
    disease_category_list = list()

    disease_index_name_dict = dict()
    feature_index_name_dict = dict()
    risk_factor_index_name_dict = dict()
    label_index_name_dict = dict()
    disease_category_index_name_dict = dict()
    treatment_index_name_dict = dict()

    general_index_category_dict = dict()
    with open(path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        feature_idx = 0
        label_idx = 0
        disease_idx = 0
        risk_factor_idx = 0
        treatment_idx = 0
        disease_category_idx = 0

        line_idx = 0
        for line in csv_reader:
            if line_idx == 0:
                line_idx += 1
                # 跳过前两个是因为前两个是p_id和v_id
                for idx in range(2, len(line)):
                    name = line[idx]
                    if name.__contains__('label'):
                        general_index_category_dict[idx] = 'label'
                        label_index_name_dict[label_idx] = name
                        label_idx += 1
                    elif name.__contains__('disease'):
                        general_index_category_dict[idx] = 'disease'
                        disease_index_name_dict[disease_idx] = name
                        disease_idx += 1
                    elif name.__contains__('treatment'):
                        general_index_category_dict[idx] = 'treatment'
                        treatment_index_name_dict[treatment_idx] = name
                        treatment_idx += 1
                    elif name.__contains__('feature'):
                        general_index_category_dict[idx] = 'feature'
                        feature_index_name_dict[feature_idx] = name
                        feature_idx += 1
                    elif name.__contains__('risk_factor'):
                        general_index_category_dict[idx] = 'risk_factor'
                        risk_factor_index_name_dict[risk_factor_idx] = name
                        risk_factor_idx += 1
                    elif name.__contains__('category'):
                        general_index_category_dict[idx] = 'disease_category'
                        disease_category_index_name_dict[disease_category_idx] = name
                        disease_category_idx += 1
                    else:
                        raise ValueError('name error')
                continue
            pat_visit_list.append([line[0], line[1]])
            label_line = list()
            feature_line = list()
            disease_line = list()
            risk_factor_line = list()
            treatment_line = list()
            disease_category_line = list()
            for idx in range(2, len(line)):
                if general_index_category_dict[idx] == 'label':
                    label_line.append(line[idx])
                elif general_index_category_dict[idx] == 'treatment':
                    treatment_line.append(line[idx])
                elif general_index_category_dict[idx] == 'risk_factor':
                    risk_factor_line.append(line[idx])
                elif general_index_category_dict[idx] == 'disease':
                    disease_line.append(line[idx])
                elif general_index_category_dict[idx] == 'feature':
                    feature_line.append(line[idx])
                elif general_index_category_dict[idx] == 'disease_category':
                    disease_category_line.append(line[idx])
                else:
                    raise ValueError('')
            feature_list.append(feature_line)
            label_list.append(label_line)
            disease_list.append(disease_line)
            risk_factor_list.append(risk_factor_line)
            treatment_list.append(treatment_line)
            disease_category_list.append(disease_category_line)
    return feature_list, label_list, disease_list, risk_factor_list, treatment_list, disease_category_list,\
        pat_visit_list, feature_index_name_dict, label_index_name_dict, disease_index_name_dict, \
        risk_factor_index_name_dict, treatment_index_name_dict, disease_category_index_name_dict


def five_fold_split(save_folder, data_length, **args):
    """20200731复核"""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    idx_list = [i for i in range(data_length)]
    random.shuffle(idx_list)

    new_data_dict = dict()
    batch_size = data_length // 5
    for key in args:
        new_data_dict[key] = [[], [], [], [], []]
        for i in range(batch_size):
            new_data_dict[key][0].append(args[key][idx_list[i]])
        for i in range(batch_size, batch_size*2):
            new_data_dict[key][1].append(args[key][idx_list[i]])
        for i in range(batch_size*2, batch_size*3):
            new_data_dict[key][2].append(args[key][idx_list[i]])
        for i in range(batch_size*3, batch_size*4):
            new_data_dict[key][3].append(args[key][idx_list[i]])
        for i in range(batch_size*4, data_length):
            new_data_dict[key][4].append(args[key][idx_list[i]])
    for key in new_data_dict:
        for index in range(len(new_data_dict[key])):
            if key == 'pat_visit_list':
                data = np.array(new_data_dict[key][index])
            else:
                data = np.array(new_data_dict[key][index], dtype=np.float)
            np.save(os.path.join(save_folder, '{}_{}.npy'.format(key, index)), data)


def main():
    """
    20200731复核
    此处的五折交叉验证主要是为基线模型准备的，并为后期的PBXAI的分法做了样板，
    由于covert_and_impute模块已经重新矫正过kg的Idx对齐，本脚本中没有出现会打乱顺序的操作，因此此处无需再进行列的idx校正
    :return:
    """
    file_path = os.path.abspath('../../resource/preprocessed_data/mimic_imputed_data.csv')
    five_fold_folder_five_part = os.path.abspath('../../resource/preprocessed_data/mimic_five_part_five_fold')
    five_fold_folder_two_part = os.path.abspath('../../resource/preprocessed_data/mimic_two_part_five_fold')

    feature_list, label_list, disease_list, risk_factor_list, treatment_list, disease_category_list, \
        pat_visit_list, feature_index_name_dict, label_index_name_dict, disease_index_name_dict, \
        risk_factor_index_name_dict, treatment_index_name_dict, disease_category_index_name_dict = \
        read_data_split_to_feature_risk_treatment_label(file_path)

    label_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_label.csv')
    risk_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_risk.csv')
    disease_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_disease.csv')
    disease_category_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_disease_category.csv')
    feature_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_feature.csv')
    treatment_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_5_part_treatment.csv')
    save_file(pat_visit_list, feature_list, feature_index_name_dict, feature_file)
    save_file(pat_visit_list, treatment_list, treatment_index_name_dict, treatment_file)
    save_file(pat_visit_list, label_list, label_index_name_dict, label_file)
    save_file(pat_visit_list, risk_factor_list, risk_factor_index_name_dict, risk_file)
    save_file(pat_visit_list, disease_list, disease_index_name_dict, disease_file)
    save_file(pat_visit_list, disease_category_list, disease_category_index_name_dict, disease_category_file)

    five_fold_split(five_fold_folder_five_part, len(feature_list), feature_list=feature_list, label_list=label_list,
                    risk_factor_list=risk_factor_list, treatment_list=treatment_list, disease_list=disease_list,
                    pat_visit_list=pat_visit_list, disease_category_list=disease_category_list,)

    feature_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_2_part_feature.csv')
    label_file = os.path.abspath('../../resource/preprocessed_data/mimic_split_2_part_label.csv')
    feature_list, label_list, pat_visit_list, label_index_name_dict, feature_index_name_dict = \
        read_data_split_to_feature_and_label(file_path)
    save_file(pat_visit_list, feature_list, feature_index_name_dict, feature_file)
    save_file(pat_visit_list, label_list, label_index_name_dict, label_file)

    five_fold_split(five_fold_folder_two_part, len(feature_list), feature_list=feature_list, label_list=label_list,
                    pat_visit_list=pat_visit_list)


if __name__ == '__main__':
    main()
