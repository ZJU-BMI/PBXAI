import os
import sys
src = os.path.abspath('../')
sys.path.append(src)
sys.path.append(os.path.join(src, 'model'))
sys.path.append(os.path.join(src, 'data_preprocess'))
sys.path.append(os.path.join(src, 'knowledge_graph'))

import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as func
import torch.optim as opt
from collections import namedtuple
from datetime import datetime
import torch
import argparse
import numpy as np

import experiment_util as util
from model import kg_env, performance_eval
import random


#
# Model V5 针对由于参数化问题产生的模型错误，首先取消LSTM的state分立设计
#
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
logger = util.get_logger(os.path.abspath('../../resource/agent/train_log_{}.txt').
                         format(datetime.now().strftime('%Y%m%d%H%M%S')))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_sizes, gamma):
        assert act_dim == 65
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(state_dim, hidden_sizes[0])
        self.l4 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        assert act_mask.shape[1] == 65
        x = func.relu(self.l1(state))
        x = func.relu(self.l2(x))
        actor_logits = self.actor(x)

        zero_idx = act_mask.clone() == 0
        actor_logits[zero_idx] = -99
        act_probs = func.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        x = func.relu(self.l3(state))
        x = func.relu(self.l4(x))
        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_act_mask, device):
        batch_hidden_state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.LongTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]

        probs, value = self((batch_hidden_state, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        # 按照我们的设计，不应该出现采样到非法点的情况，如果出现了，就不满足下面的断言，就报错
        # assert valid_idx.sum() == acts.shape[0]

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight):
        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        # loss = critic_loss
        # loss = actor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


class ACDataLoader(object):
    def __init__(self, p_v_id_list, batch_size):
        self.p_v_id = np.array(p_v_id_list)
        self.num_visit = len(p_v_id_list)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_visit)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            raise ValueError('no next batch')
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_visit)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        self._has_next = self._has_next and end_idx < self.num_visit
        self._start_idx = end_idx
        return batch_idx.tolist()


def read_patient_representation_and_label(data_source, info_folder, embed_folder, test_idx, data_fraction=1,
                                          raw_data=False, omit_duplicate_disease=False):
    if test_idx not in {1, 2, 3, 4, 0}:
        raise ValueError('Error Test Index')

    train_pat_embed = list()
    train_label = list()
    train_disease = list()
    train_risk_factor = list()
    train_category = list()
    train_id = list()

    for i in range(5):
        if i != test_idx:
            if raw_data:
                train_pat_embed.append(
                    np.load(os.path.join(embed_folder, '{}_pat_repre_raw_{}.npy'.format(data_source, i))))
            else:
                train_pat_embed.append(
                    np.load(os.path.join(embed_folder, '{}_pat_representation_{}.npy'.format(data_source, i))))
            train_label.append(np.load(os.path.join(info_folder, 'label_list_{}.npy'.format(i))))
            train_disease.append(np.load(os.path.join(info_folder, 'disease_list_{}.npy'.format(i))))
            train_risk_factor.append(np.load(os.path.join(info_folder, 'risk_factor_list_{}.npy'.format(i))))
            train_category.append(np.load(os.path.join(info_folder, 'disease_category_list_{}.npy'.format(i))))
            train_id.append(np.load(os.path.join(info_folder, 'pat_visit_list_{}.npy'.format(i))))

    train_pat_embed = np.concatenate(train_pat_embed, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_disease = np.concatenate(train_disease, axis=0)
    train_risk_factor = np.concatenate(train_risk_factor, axis=0)
    train_category = np.concatenate(train_category, axis=0)
    train_id = np.concatenate(train_id, axis=0)

    if omit_duplicate_disease:
        train_label = train_label - train_label * train_disease
    print(np.sum(train_label)/len(train_label))

    # 按照规矩，先Risk factor，再disease, 再category
    # 在label中，抹掉risk factor category之类的信息，只计算disease命中率
    train_interact = np.concatenate([train_risk_factor, train_disease, train_category], axis=1)
    train_label = np.concatenate([np.zeros(train_risk_factor.shape), train_label, np.zeros(train_category.shape)],
                                 axis=1)
    if raw_data:
        test_pat_embed = np.load(os.path.join(embed_folder, '{}_pat_repre_raw_{}.npy'.format(data_source, test_idx)))
    else:
        test_pat_embed = np.load(os.path.join(
            embed_folder, '{}_pat_representation_{}.npy'.format(data_source, test_idx)))
    test_label = np.load(os.path.join(info_folder, 'label_list_{}.npy'.format(test_idx)))
    test_disease = np.load(os.path.join(info_folder, 'disease_list_{}.npy'.format(test_idx)))
    test_risk_factor = np.load(os.path.join(info_folder, 'risk_factor_list_{}.npy'.format(test_idx)))
    test_category = np.load(os.path.join(info_folder, 'disease_category_list_{}.npy'.format(test_idx)))
    test_id = np.load(os.path.join(info_folder, 'pat_visit_list_{}.npy'.format(test_idx)))
    test_interact = np.concatenate([test_risk_factor, test_disease, test_category], axis=1)

    if omit_duplicate_disease:
        test_label = test_label - test_label * test_disease

    test_label = np.concatenate([np.zeros(test_risk_factor.shape), test_label, np.zeros(test_category.shape)], axis=1)

    index = [i for i in range(len(train_label))]
    random.shuffle(index)
    index = index[: int(len(train_label) * data_fraction)]
    train_pat_embed = train_pat_embed[index]
    train_id = train_id[index]
    train_label = train_label[index]
    train_interact = train_interact[index]

    return train_pat_embed, train_interact, train_label, train_id, test_pat_embed, test_interact, test_label, test_id


def train(args):
    pat_embed, interact, label, pat_id, _, _, _, _ = read_patient_representation_and_label(
        args.data_source, info_folder=args.data_path, embed_folder=args.pat_representation_folder,
        test_idx=args.test_fold_idx, omit_duplicate_disease=args.omit_duplicate)
    pat_idx_list = [i for i in range(len(pat_embed))]

    env = kg_env.BatchKGEnvironment(args.kg_path, args.embed_path, args.max_acts, args.max_path_len, len(pat_embed[0]),
                                    args.history_len)

    data_loader = ACDataLoader(pat_idx_list, args.batch_size)
    model = ActorCritic(env.state_dim, env.max_acts, args.hidden, args.gamma).to(args.device)
    optimizer = opt.Adam(model.parameters(), lr=args.lr)

    policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.save_path, 0)
    torch.save(model.state_dict(), policy_file)

    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_losses, total_p_losses, total_v_losses, total_entropy, total_rewards = [], [], [], [], []
        # Start epoch
        data_loader.reset()
        while data_loader.has_next():
            batch_id = data_loader.get_batch()
            batch_embed = pat_embed[batch_id]
            batch_interact = interact[batch_id]
            batch_label = label[batch_id]
            batch_pat_id = pat_id[batch_id]
            # Start batch episodes
            batch_state = env.reset(batch_id, batch_embed, batch_interact)
            done = False
            while not done:
                batch_act_mask = env.batch_action_mask()
                batch_act = model.select_action(batch_state, batch_act_mask, args.device)  # int
                batch_state, batch_reward, done = env.batch_step(batch_act, batch_embed, batch_label)
                # 每次update函数执行后，都会重置rewards, entropy和saved actions函数，因此此处不用担心累积
                model.rewards.append(batch_reward)
            # End of episodes

            # 用于统计，由于reward会在update后重置，因此要在此处append
            total_rewards.append(np.sum(model.rewards))

            # Update policy
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(pat_idx_list) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            loss, p_loss, v_loss, e_loss = model.update(optimizer, args.device, args.ent_weight)

            total_losses.append(loss)
            total_p_losses.append(p_loss)
            total_v_losses.append(v_loss)
            total_entropy.append(e_loss)
            step += 1

        avg_reward = np.mean(total_rewards) / args.batch_size
        avg_loss = np.mean(total_losses)
        avg_p_loss = np.mean(total_p_losses)
        avg_v_loss = np.mean(total_v_losses)
        avg_entropy = np.mean(total_entropy)
        logger.info(
                'epoch={:d}'.format(epoch) +
                ' | loss={:.5f}'.format(avg_loss) +
                ' | p loss={:.5f}'.format(avg_p_loss) +
                ' | v loss={:.5f}'.format(avg_v_loss) +
                ' | entropy={:.5f}'.format(avg_entropy) +
                ' | reward={:.5f}'.format(avg_reward))
        # END of epoch
        policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.save_path, epoch)
        torch.save(model.state_dict(), policy_file)


def main():
    max_act = 65
    max_path_len = 2
    hidden = [64, 32]
    epoch = 10
    gamma = 0.1
    ent_weight = 0.13
    learning_rate = 0.001
    top_k = [10, 5, 5, 2, 2]
    batch_size = 64
    history_len = 1
    data_source = 'mimic'  # mimic plagh

    for test_fold_idx in [0, 1, 2, 3, 4]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
        parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
        parser.add_argument('--test_fold_idx', type=int, default=test_fold_idx)
        parser.add_argument('--omit_duplicate', type=bool, default=False)
        parser.add_argument('--epochs', type=int, default=epoch)
        parser.add_argument('--batch_size', type=int, default=batch_size)
        parser.add_argument('--history_len', type=int, default=history_len)
        parser.add_argument('--data_source', type=str, default=data_source)
        parser.add_argument('--lr', type=float, default=learning_rate)
        parser.add_argument('--max_acts', type=int, default=max_act, help='Max number of actions.')
        parser.add_argument('--max_path_len', type=int, default=max_path_len, help='Max path length.')
        parser.add_argument('--gamma', type=float, default=gamma, help='reward discount factor.')
        parser.add_argument('--ent_weight', type=float, default=ent_weight, help='weight factor for entropy loss')
        parser.add_argument('--hidden', type=int, nargs='*', default=hidden, help='number of samples')
        parser.add_argument('--kg_path', type=str, default=os.path.abspath('../../resource/knowledge_graph/kg.pkl'))
        parser.add_argument('--embed_path', type=str,
                            default=os.path.abspath('../../resource/representation/{}_medical_concept_embedding.npy'
                                                    .format(data_source)))
        parser.add_argument('--pat_representation_folder', type=str,
                            default=os.path.abspath('../../resource/representation/'))
        parser.add_argument('--data_path', type=str, default=os.path.abspath(
            '../../resource/preprocessed_data/{}_five_part_five_fold'.format(data_source)))
        parser.add_argument('--save_path', type=str, default=os.path.abspath('../../resource/agent/'))
        parser.add_argument('--run_path', default=True)
        parser.add_argument('--run_eval', default=True, help='Run evaluation?')
        parser.add_argument('--topk', type=int, nargs='*', default=top_k, help='number of samples')
        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        logger.info(args)
        train(args)
        performance_eval.test(args, 'test')
        # performance_eval.test(args, 'train')


if __name__ == '__main__':
    main()
