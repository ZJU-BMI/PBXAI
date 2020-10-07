import torch
import numpy as np
import os


def main():
    """
    使用RBM参数矩阵作为知识图谱中各个节点的Concept Embedding Representation
    :return:
    """
    num_visible = 65
    num_hidden = 5
    batch_size = 32
    cuda = torch.cuda.is_available()
    cd_k = 2
    epoch = 30
    data_source = 'mimic'
    if cuda:
        torch.cuda.set_device(0)

    file_folder = os.path.abspath('../../resource/preprocessed_data/{}_five_part_five_fold'.format(data_source))
    data_loader = DataLoader(file_folder, batch_size)

    rbm = RestrictedBoltzmannMachine(num_visible, num_hidden, cd_k, use_cuda=cuda)
    for epoch in range(epoch):
        epoch_error = 0.0
        data_list_in_batch = data_loader.get_batch_list()

        for batch in data_list_in_batch:
            batch = torch.from_numpy(batch).float()
            if cuda:
                batch = batch.cuda()
            batch_error = rbm.contrastive_divergence(batch)
            epoch_error += batch_error
        print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

    save_path = os.path.abspath('../../resource/representation/{}_medical_concept_embedding.npy'.format(data_source))
    weight_mat = rbm.weights.cpu().data.numpy()
    np.save(save_path, weight_mat)


class DataLoader(object):
    """20200715复核"""
    def __init__(self, file_folder, batch_size):
        """由于知识图谱中的编号顺序是risk, disease, category，因此此处也以这个顺序进行合并"""
        data_list = list()
        for i in range(5):
            risk_factor = np.load(os.path.join(file_folder, 'risk_factor_list_{}.npy'.format(i)))
            disease = np.load(os.path.join(file_folder, 'disease_list_{}.npy'.format(i)))
            disease_category = np.load(os.path.join(file_folder, 'disease_category_list_{}.npy'.format(i)))
            data_list.append(np.concatenate([risk_factor, disease, disease_category], axis=1))
        self._data = np.concatenate([data_list[0], data_list[1], data_list[2], data_list[3], data_list[4]], axis=0)
        self._batch_size = batch_size

    def get_batch_list(self):
        batch_num = len(self._data) // self._batch_size
        batch_data_list = list()
        idx_permutation = [i for i in range(len(self._data))]
        np.random.shuffle(idx_permutation)
        for i in range(batch_num):
            idx_list = np.array(idx_permutation[i*self._batch_size: (i+1)*self._batch_size])
            batch_data_list.append(np.array(self._data[idx_list], dtype=np.float))
        return batch_data_list


class RestrictedBoltzmannMachine(object):

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        visible_probabilities = None
        hidden_probabilities = None
        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


if __name__ == '__main__':
    main()

