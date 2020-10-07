from itertools import islice
import csv
import os
import pickle
from experiment_util import DISEASE, DISEASE_CATEGORY, RISK_FACTOR, SUB_CONCEPT, GENERAL_CONCEPT, CAUSE, \
    relation_list, entity_type_list, kg_relation
"""
20200720复核
20200822复核
"""


class KnowledgeGraph(object):
    def __init__(self, data_dir):
        self._data_dir = data_dir
        # create knowledge graph
        self.G = dict()
        self.idx_entity_type_dict = dict()
        # load data
        self._data = {DISEASE_CATEGORY: [], DISEASE: [],  RISK_FACTOR: [], SUB_CONCEPT: [], GENERAL_CONCEPT: [],
                      CAUSE: []}
        self._load_data()
        self._load_entities()
        self._load_relations()
        self._clean()

        # for use
        self.top_matches = None
        self.degrees = {}
        self.compute_degrees()
        self.index_valid_test()

    def index_valid_test(self):
        # 我们要求所有的entity拥有全局唯一的id
        index_set = set()
        for entity_type in self.G:
            for index in self.G[entity_type]:
                if index_set.__contains__(index):
                    raise ValueError('Duplicate item index')
                else:
                    index_set.add(index)
        for entity_type in self.G:
            for index in self.G[entity_type]:
                for type_2 in self.G[entity_type][index]:
                    for index_2 in self.G[entity_type][index][type_2]:
                        if not (index_set.__contains__(index_2) and index_set.__contains__(index)):
                            raise ValueError('Error item index')

    def get_index_type(self, index):
        return self.idx_entity_type_dict[int(index)]

    def _load_data(self):
        with open(self._data_dir, 'r', encoding='utf-8-sig', newline='') as file:
            csv_reader = csv.reader(file)
            for line in islice(csv_reader, 1, None):
                idx, label, chinese_name, english_name, start_idx, end_idx, relation_type = \
                    line[0], line[1], line[2], line[4], line[5], line[6], line[7]
                if len(label) == 0:
                    if relation_type == 'GENERAL_CONCEPT':
                        self._data[GENERAL_CONCEPT].append([int(start_idx), int(end_idx), GENERAL_CONCEPT])
                    elif relation_type == 'SUB_CONCEPT':
                        self._data[SUB_CONCEPT].append([int(start_idx), int(end_idx), SUB_CONCEPT])
                    elif relation_type == "CAUSE":
                        self._data[CAUSE].append([int(start_idx), int(end_idx), CAUSE])
                    else:
                        raise ValueError('Error')
                else:
                    if label == ':DISEASE_CATEGORY':
                        self._data[DISEASE_CATEGORY].append([int(idx), label, chinese_name, english_name])
                        self.idx_entity_type_dict[int(idx)] = DISEASE_CATEGORY
                    elif label == ':DISEASE':
                        self._data[DISEASE].append([int(idx), label, chinese_name, english_name])
                        self.idx_entity_type_dict[int(idx)] = DISEASE
                    elif label == ':RISK_FACTOR':
                        self._data[RISK_FACTOR].append([int(idx), label, chinese_name, english_name])
                        self.idx_entity_type_dict[int(idx)] = RISK_FACTOR
                    else:
                        raise ValueError('Error Label Name')
        print('load knowledge graph success')

    def _load_entities(self):
        print('load entities...')
        # 注意 entity的编号全局唯一，也就是不同种类的entity在index上也不存在交叉，根据任意index就可以追溯到entity
        num_nodes = 0
        for entity in entity_type_list:
            self.G[entity] = dict()
            entity_size = len(self._data[entity])
            for eid in range(entity_size):
                self.G[entity][self._data[entity][eid][0]] = {r: [] for r in kg_relation[entity]}
            num_nodes += entity_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_relations(self):
        # 注意，此处不建模自环
        for relation in relation_list:
            data = self._data[relation]
            for idx in range(len(data)):
                start_idx, end_idx, _ = data[idx]
                if start_idx == end_idx:
                    continue
                start_type = self.idx_entity_type_dict[start_idx]
                self._add_edge(start_type, start_idx, relation, end_idx)
        print('load relations')

    def _add_edge(self, etype1, eid1, relation, eid2):
        self.G[etype1][eid1][relation].append(eid2)

    def _clean(self):
        print('Remove duplicates...')
        count = 0
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    original_len = len(self.G[etype][eid][r])
                    data = set(self.G[etype][eid][r])
                    if data.__contains__(eid):
                        data.remove(eid)
                    data = sorted(list(data))
                    count += (original_len - len(data))
                    self.G[etype][eid][r] = data
        print('remove duplicate relation count: {}'.format(count))

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        relation_sum = 0
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count
                relation_sum += count
        max_degree = 0
        for i in self.degrees:
            for j in self.degrees[i]:
                if max_degree < self.degrees[i][j]:
                    max_degree = self.degrees[i][j]
        print('max degree: {}'.format(max_degree))
        print('relation sum: {}'.format(relation_sum))

    def get(self, eh_type, eh_id=None, relation=None):
        """get 这一步需要在KG具体被调用时判断正确性"""
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)


def main():
    file_path = os.path.abspath('../../resource/knowledge_graph/knowledge_graph_output.csv')
    knowledge = KnowledgeGraph(file_path)
    save_path = os.path.abspath('../../resource/knowledge_graph/kg.pkl')
    pickle.dump(knowledge, open(save_path, 'wb'))


if __name__ == '__main__':
    main()
