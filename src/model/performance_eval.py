from __future__ import absolute_import, division, print_function
import csv
from model.kg_env import BatchKGEnvironment
from model.train_agent import *
import numpy as np
import torch
import experiment_util as util


def acts_pool_to_mask(act_pool):
    act_mask_list = []
    for candidates in act_pool:
        mask = np.zeros([65])
        for act in candidates:
            idx = act[1]
            assert 0 <= idx < 65
            mask[idx] = 1
        act_mask_list.append(mask)
    mask = np.vstack(act_mask_list)
    return mask


def batch_beam_search(env, model, test_pat_embed, test_interact, batch_pat_ids, max_len, device, topk):
    # id embedding mapping, 由于可选path会增长，因此需要构建合适的映射，在必要的时候映射回其idx
    id_pat_dict = dict()
    for idx in range(len(batch_pat_ids)):
        id_pat_dict[batch_pat_ids[idx] * -1 - 10000] = idx
    model.eval()

    state_pool = env.reset(batch_pat_ids, test_pat_embed, test_interact)
    path_pool = env.get_batch_path()  # list of list, size=bs
    probs_pool = [[] for _ in batch_pat_ids]
    for hop in range(max_len):
        state_tensor = torch.from_numpy(state_pool).float()
        # 此处获得的是每个batch对应的node的可选action列表（可选Action数量已经被限制在max_act范围内）
        acts_pool = env.batch_get_actions(path_pool, False, test_interact)
        # 测试阶段不做mask，此处仅仅是点出存在的act
        act_mask_pool = acts_pool_to_mask(acts_pool)
        act_mask_tensor = torch.from_numpy(act_mask_pool).clone().long().to(device)
        # 此处获得的是单步每个动作的prob，然后找出顺位最高的几个
        probs, _ = model((state_tensor, act_mask_tensor))  # Tensor of [bs, act_dim]
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().clone().cpu().numpy()
        topk_probs = topk_probs.detach().clone().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        reverse_dict = dict()
        new_pool_idx = 0
        for row in range(len(topk_idxs)):
            path = path_pool[row]
            probs = probs_pool[row]
            for global_idx, p in zip(topk_idxs[row], topk_probs[row]):
                if act_mask_pool[row][global_idx] == 0:
                    # 当遇到非法路径时跳过
                    continue
                act_idx = -9999999
                reverse_dict[new_pool_idx] = row
                new_pool_idx += 1

                assert 0 <= global_idx < 65
                for idx in range(len(acts_pool[row])):
                    if global_idx == acts_pool[row][idx][1]:
                        assert act_idx == -9999999
                        act_idx = idx
                assert act_idx != -9999999

                relation, next_node_id = acts_pool[row][act_idx]  # (relation, next_node_id)
                if relation == util.SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = env.kg.get_index_type(next_node_id)
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool

        state_pool = env.batch_get_state(path_pool, test_pat_embed, id_pat_dict)
    return path_pool, probs_pool


def predict_paths(policy_file, test_pat_embed, test_interact, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.kg_path, args.embed_path, args.max_acts, args.max_path_len, len(test_pat_embed[0]),
                             args.history_len)
    pre_train_model = torch.load(policy_file)

    model = ActorCritic(env.state_dim, env.max_acts, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pre_train_model)
    model.load_state_dict(model_sd)
    model.eval()

    test_pat_ids = [i for i in range(len(test_interact))]

    start_idx = 0
    all_paths, all_probs = [], []
    while start_idx < len(test_pat_ids):
        print('current index: {}'.format(start_idx))
        end_idx = min(start_idx + args.batch_size, len(test_pat_ids))
        batch_id = test_pat_ids[start_idx:end_idx]
        batch_interact = test_interact[batch_id]
        batch_embed = test_pat_embed[batch_id]
        paths, probs = batch_beam_search(env, model, batch_embed, batch_interact, batch_id, args.max_path_len,
                                         args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
    predicts = {'paths': all_paths, 'probs': all_probs}
    return predicts


def read_group(data_path, omit):
    group = dict()
    all_label = list()
    for i in range(5):
        label = np.array(np.load(os.path.join(data_path, 'label_list_{}.npy'.format(i))), dtype=int)
        disease = np.array(np.load(os.path.join(data_path, 'disease_list_{}.npy'.format(i))), dtype=int)
        label_list = list()
        for j in range(len(label)):
            single_line_1 = list()
            single_line_2 = list()
            for k in range(len(label[0])):
                single_line_1.append(label[j][k])
                single_line_2.append(disease[j][k])
            if omit:
                line_1, line_2 = np.array(single_line_1), np.array(single_line_2)
                label_list.append(line_1 - line_1 * line_2)
            else:
                label_list.append(single_line_1)
        all_label.append(np.array(label_list))

    exclude_set = set()
    for i in range(len(all_label)):
        label_list = all_label[i]
        sum_less_than_5 = np.sum(label_list, axis=0) < 5
        for idx, item in enumerate(sum_less_than_5):
            if item:
                exclude_set.add(idx)
    all_label = np.concatenate(all_label, axis=0)
    label_sum = np.sum(all_label, axis=0)
    label_count = [(idx, item) for idx, item in enumerate(label_sum)]
    label_count = sorted(label_count, key=lambda x: x[1], reverse=True)
    label_idx_list = list()
    for item in label_count:
        if item[0] not in exclude_set:
            label_idx_list.append(item[0])

    group = {'all': [], 'group_0': [], 'group_1': [], 'group_2': []}
    for i, item in enumerate(label_idx_list):
        group['all'].append(item)
        if i < len(label_idx_list) / 3:
            group['group_0'].append(item)
        elif i < 2 * len(label_idx_list) / 3:
            group['group_1'].append(item)
        else:
            group['group_2'].append(item)

    return group


def test(args, mode):
    policy_file = os.path.abspath('../../resource/agent/policy_model_epoch_{}.ckpt'.format(args.epochs))
    path = os.path.abspath('../../resource/path_predicts_{}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S')))

    train_pat_embed, train_interact, train_label, train_id, test_pat_embed, test_interact, test_label, test_id = \
        read_patient_representation_and_label(info_folder=args.data_path,
                                              embed_folder=args.pat_representation_folder,
                                              test_idx=args.test_fold_idx, data_source=args.data_source,
                                              omit_duplicate_disease=args.omit_duplicate)
    group = read_group(args.data_path, omit=args.omit_duplicate)
    if args.run_path:
        if mode == 'test':
            predicts = predict_paths(policy_file, test_pat_embed, test_interact, args)
            performance_evaluation(predicts, test_label, args.test_fold_idx, group, args, mode)
            save_paths(predicts, path)
        elif mode == 'train':
            predicts = predict_paths(policy_file, train_pat_embed, train_interact, args)
            performance_evaluation(predicts, train_label, args.test_fold_idx, group, args, mode)
            save_paths(predicts, path)
        else:
            raise ValueError('')


def save_paths(predicts, save_path):
    data_to_write = []
    for idx in range(len(predicts['paths'])):
        paths = predicts['paths'][idx]
        probs = predicts['probs'][idx]
        line_1 = []
        line_2 = ['']
        line_3 = []
        for item in paths:
            line_1.append(item)
        for item in probs:
            line_2.append(item)
        data_to_write.append(line_1)
        data_to_write.append(line_2)
        data_to_write.append(line_3)
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def performance_evaluation(predicts_list, label, data_type, group, args, mode, cut_idx=(7, 60)):
    score = np.zeros(label.shape)
    for idx in range(len(predicts_list['paths'])):
        path = predicts_list['paths'][idx]
        prob_list = predicts_list['probs'][idx]
        pat_idx = (path[0][2]+10000)*-1
        assert pat_idx >= 0
        if path[-1][1] != util.DISEASE:
            continue
        disease_idx = path[-1][2]
        assert 7 <= disease_idx < 60
        prob = 1
        for item in prob_list:
            prob *= item
        assert 0 <= prob <= 1
        score[pat_idx, disease_idx] += prob
    score = score[:, cut_idx[0]: cut_idx[1]]
    score_sum = np.sum(score, axis=1) + 0.000000000001
    pred = (score.transpose() / score_sum).transpose()
    print(np.sum(pred, axis=1))
    label = label[:, cut_idx[0]: cut_idx[1]]
    print(np.sum(label))

    data_to_write = []
    for arg in vars(args):
        data_to_write.append([arg, getattr(args, arg)])

    head = ['model_name', 'fold', 'group', 'train/test', 'macro_auc', 'micro_auc', 'micro_f1', 'macro_f1',
            'micro_avg_precision', 'macro_avg_precision', 'coverage', 'ranking_loss', 'hamming', 'top_1_num',
            'top_3_num', 'top_5_num', 'top_10_num', 'top_20_num', 'top_30_num', 'top_40_num', 'top_50_num']
    data_to_write.append(head)

    if mode == 'test':
        test_fold_idx = args.test_fold_idx
    else:
        test_fold_idx = '/'
    for key in group:
        macro_auc, micro_auc, micro_f1, macro_f1, micro_avg_precision, macro_avg_precision, coverage, \
            ranking_loss, hamming, top_1_num, top_3_num, top_5_num, top_10_num, top_20_num, top_30_num, top_40_num,\
            top_50_num = util.metric(pred.transpose(), label.transpose(), group[key])
        data_to_write.append(['PBXAI', test_fold_idx, key, data_type, macro_auc, micro_auc, micro_f1, macro_f1,
                              micro_avg_precision, macro_avg_precision, coverage, ranking_loss, hamming, top_1_num,
                              top_3_num, top_5_num, top_10_num, top_20_num, top_30_num, top_40_num, top_50_num])
    with open(os.path.abspath('../../resource/pbxai_result_{}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S'))),
              'w', encoding='utf-8-sig', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def main():
    """V5"""
    max_acts = 65
    max_len = 0
    gamma = 0
    hidden = [32, 16]
    hidden_state_size = 8
    test_fold_idx = 4
    batch_size = 256
    top_k = [23, 23, 23]
    epochs = [1]
    mode = 'test'

    for epoch in epochs:
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
        parser.add_argument('--test_fold_idx', type=int, default=test_fold_idx, help='Max number of epochs.')
        parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
        parser.add_argument('--batch_size', type=int, default=batch_size, help='Max number of actions.')
        parser.add_argument('--epochs', type=int, default=epoch, help='Max number of actions.')
        parser.add_argument('--max_acts', type=int, default=max_acts, help='Max number of actions.')
        parser.add_argument('--max_path_len', type=int, default=max_len, help='Max path length.')
        parser.add_argument('--gamma', type=float, default=gamma, help='reward discount factor.')
        parser.add_argument('--hidden_state_size', type=int, default=hidden_state_size, help='state history length')
        parser.add_argument('--hidden', type=int, nargs='*', default=hidden, help='number of samples')
        parser.add_argument('--run_path', default=True, help='Generate predicted path? (takes long time)')
        parser.add_argument('--run_eval', default=True, help='Run evaluation?')
        parser.add_argument('--topk', type=int, nargs='*', default=top_k, help='number of samples')
        parser.add_argument('--pat_representation_folder', type=str,
                            default=os.path.abspath('../../resource/representation/'))
        parser.add_argument('--data_path', type=str, default=os.path.abspath(
            '../../resource/preprocessed_data/plagh_five_part_five_fold'))
        parser.add_argument('--kg_path', type=str, default=os.path.abspath('../../resource/knowledge_graph/kg.pkl'))
        parser.add_argument('--embed_path', type=str,
                            default=os.path.abspath('../../resource/representation/medical_concept_embedding.npy'))
        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        test(args, mode=mode)


if __name__ == '__main__':
    main()
