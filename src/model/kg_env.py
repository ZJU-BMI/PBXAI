from experiment_util import load_kg, load_embed, relation_embed, GENERAL_CONCEPT, PATIENT, SELF_LOOP, HAVE, DISEASE
import numpy as np


class KGState(object):
    def __init__(self, patient_size, concept_size, relation_size, history_len=1):
        self.history_len = history_len
        self.patient_size = patient_size
        self.concept_size = concept_size
        self.relation_size = relation_size
        if patient_size != concept_size:
            raise ValueError('')
        if history_len == 0:
            self.dim = patient_size + concept_size
        elif history_len == 1:
            self.dim = patient_size + concept_size * 2 + relation_size
        elif history_len == 2:
            self.dim = patient_size + 3 * concept_size + 2 * relation_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, patient, curr_node, last_node, last_relation, older_node, older_relation):
        if self.history_len == 0:
            return np.concatenate([patient, curr_node])
        elif self.history_len == 1:
            return np.concatenate([patient, curr_node, last_node, last_relation])
        elif self.history_len == 2:
            return np.concatenate([patient, curr_node, last_node, last_relation, older_node, older_relation])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    """V5"""
    def __init__(self, kg_path, embed_path, max_acts, max_path_len, pat_repre_size, history=2):
        # 规定max acts即为知识图谱中的所有可达节点
        assert max_acts == 65
        self.max_acts = max_acts
        self.max_len = max_path_len
        # 因为最初始的点也占了1，因此最大path长度应当是max_path_len+1
        self.max_num_nodes = max_path_len + 1
        self.kg = load_kg(kg_path)
        self.embeds = load_embed(embed_path)
        self.embed_size = self.embeds.shape[1]
        self.state_gen = KGState(pat_repre_size, self.embed_size, relation_embed[GENERAL_CONCEPT].shape[0], history)
        self.state_dim = self.state_gen.dim

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False

    def _get_actions(self, path, done, pat_init_interact=None):
        """
        Compute actions for current node.
        由于我们的图谱非常的小，因此可以不进行Action的剪枝
        为避免idx冲突，所有患者的id全部*-1再减一处理(确保一定是负数)
        """
        _, curr_node_type, curr_node_id = path[-1]
        actions = []

        # 对所有节点（除非current id小于0，也就是疾病节点,均添加自环节点）
        if not curr_node_type == PATIENT:
            assert len(path) > 1
            actions.append((SELF_LOOP, curr_node_id))

        # (1) If game is finished, only return self-loop action.
        if done:
            return actions

        if curr_node_type == PATIENT:
            assert len(path) == 1
            assert len(pat_init_interact) == 65
            relations_nodes = {HAVE: []}
            for idx_ in range(len(pat_init_interact)):
                if pat_init_interact[idx_] == 1:
                    relations_nodes[HAVE].append(idx_)
        else:
            relations_nodes = self.kg(curr_node_type, curr_node_id)

        # (2) Get all possible edges from original knowledge graph. must remove visited nodes!
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set(v[2] for v in path)
        for r in relations_nodes:
            next_node_ids = relations_nodes[r]
            ids = []
            for n in next_node_ids:
                if n not in visited_nodes:
                    ids.append(n)
            candidate_acts.extend(zip([r] * len(ids), ids))

        candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def batch_get_actions(self, batch_path, done, patient_interact_list):
        return self._batch_get_actions(batch_path, done, patient_interact_list)

    def get_batch_path(self):
        return self._batch_path

    def _batch_get_actions(self, batch_path, done, pat_interact=None):
        if len(batch_path[0]) == 1:
            return [self._get_actions(batch_path[idx_], done, pat_interact[idx_]) for idx_ in range(len(batch_path))]
        else:
            return [self._get_actions(batch_path[idx_], done) for idx_ in range(len(batch_path))]

    def _get_state(self, path, pat_embed):
        node_zero = np.zeros(self.state_gen.concept_size)
        relation_zero = np.zeros(self.state_gen.relation_size)
        if len(path) == 1:  # initial state
            state = self.state_gen(pat_embed, pat_embed, node_zero, relation_zero, node_zero, relation_zero)
            return state

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == PATIENT:
            raise ValueError('')
        else:
            assert curr_node_id >= 0
            curr_node_embed = self.embeds[curr_node_id]
        if last_node_type == PATIENT:
            if len(path) == 2:
                last_node_embed = pat_embed
            else:
                raise ValueError('')
        else:
            assert last_node_id >= 0
            last_node_embed = self.embeds[last_node_id]
        last_relation_embed = relation_embed[last_relation]
        if len(path) == 2:
            state = self.state_gen(pat_embed, curr_node_embed, last_node_embed, last_relation_embed, node_zero,
                                   relation_zero)
            return state

        _, older_node_type, older_node_id = path[-3]
        if older_node_type == PATIENT:
            older_node_embed = pat_embed
        else:
            assert last_node_id >= 0
            older_node_embed = self.embeds[older_node_id]
        older_relation_embed = relation_embed[older_relation]
        state = self.state_gen(pat_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    def batch_get_state(self, batch_path, pat_embed_list, id_embed_dict=None):
        return self._batch_get_state(batch_path, pat_embed_list, id_embed_dict)

    def _batch_get_state(self, batch_path, pat_embed_list, id_embed_dict=None):
        if id_embed_dict is not None:
            # 仅用于测试，在测试集中，由于batch_path的长度会超过pat_embed_list，因此需要进行映射
            # 在训练时，这两个长度严格相等，因此无所谓
            batch_state = []
            for idx in range(len(batch_path)):
                pat_idx = batch_path[idx][0][2]
                embed_idx = id_embed_dict[pat_idx]
                batch_state.append(self._get_state(batch_path[idx], pat_embed_list[embed_idx]))
        else:
            # 用于训练和测试的reset部分
            batch_state = [self._get_state(batch_path[idx], pat_embed_list[idx]) for idx in range(len(batch_path))]
        return np.vstack(batch_state)

    def _get_reward(self, path, label=None):
        # If it is initial state or 1-hop search, reward is 0.
        # 此处由于第一步没跳的时候也占了一位，所以path长度应当是必须大于max len才终止
        target_score = 0
        if len(path) <= self.max_len:
            return target_score

        assert len(path) == self.max_num_nodes and label is not None

        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == DISEASE:
            if label[curr_node_id] > 0.5:
                target_score = 1
            else:
                target_score = 0
        else:
            target_score = -1
        return target_score

    def _batch_get_reward(self, batch_path, label=None):
        if label is not None:
            batch_reward = [self._get_reward(batch_path[idx_], label[idx_]) for idx_ in range(len(batch_path))]
        else:
            batch_reward = [self._get_reward(batch_path[idx_]) for idx_ in range(len(batch_path))]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, pat_idx_list, pat_embedding_list, interact_list):
        # 此处，embedding_list的作用是在reset时重置pat_embedding
        # disease_list的作用是建立user和知识图谱之间的关联
        # each element is a tuple of (relation, entity_type, entity_id)
        # 为避免语义歧义，此处所有的pat_id的idx全部做取反再减10000处理，以保证区间和embed concept不同
        self._batch_path = [[(SELF_LOOP, PATIENT, pat_id * -1 - 10000)] for pat_id in pat_idx_list]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path, pat_embedding_list)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done, interact_list)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx, embed, label):
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            global_act_idx = batch_act_idx[i]
            act_idx = -9999999
            assert 0 <= global_act_idx < 65
            for idx_ in range(len(self._batch_curr_actions[i])):
                if global_act_idx == self._batch_curr_actions[i][idx_][1]:
                    # 按照设计这一过程只能命中一次，但是不可以不命中
                    assert act_idx == -9999999
                    act_idx = idx_
            assert act_idx != -9999999
            _, curr_node_type, _ = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]

            if relation == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = self.kg.get_index_type(next_node_id)
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path, embed)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path, label)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self):
        # 返回全局语义的action mask
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_mask = np.zeros(self.max_acts)
            for item in actions:
                act_mask[item[1]] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)


def main():
    print('')


if __name__ == '__main__':
    main()