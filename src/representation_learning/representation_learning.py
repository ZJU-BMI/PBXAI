import os
import numpy as np
from torch import nn, optim
import torch
from torch.nn import functional as func
"""
20200715复核
20200731复核，将representation模块换为autoencoder
"""


class DataLoader(object):
    def __init__(self, file_folder_, batch_size_):
        """
        注意，由于此处使用BCE损失，因此所有数据要被事先压缩到0-1之间
        :param file_folder_:
        :param batch_size_:
        """
        data_list = list()
        for i in range(5):
            feature = np.load(os.path.join(file_folder_, 'feature_list_{}.npy').format(i))
            treatment = np.load(os.path.join(file_folder_, 'treatment_list_{}.npy').format(i))
            risk_factor = np.load(os.path.join(file_folder_, 'risk_factor_list_{}.npy'.format(i)))
            disease = np.load(os.path.join(file_folder_, 'disease_list_{}.npy'.format(i)))
            disease_category = np.load(os.path.join(file_folder_, 'disease_category_list_{}.npy'.format(i)))
            data_list.append(np.concatenate([feature, treatment, risk_factor, disease, disease_category], axis=1))
        self._data = np.concatenate([data_list[0], data_list[1], data_list[2], data_list[3], data_list[4]], axis=0)
        self._data = np.array(self._data, dtype=np.float)

        max_value = np.max(self._data, axis=0)[np.newaxis, :] + 0.0001
        min_value = np.min(self._data, axis=0)[np.newaxis, :] - 0.0001
        self._data = (self._data - min_value) / (max_value - min_value)

        self._batch_size = batch_size_
        self.length = len(self._data)

    def get_batch_list(self):
        batch_num = len(self._data) // self._batch_size
        batch_data_list = list()
        idx_permutation = [i for i in range(len(self._data))]
        np.random.shuffle(idx_permutation)
        for i in range(batch_num):
            idx_list = np.array(idx_permutation[i*self._batch_size: (i+1)*self._batch_size])
            batch_data_list.append(np.array(self._data[idx_list], dtype=np.float))
        return batch_data_list

    def get_data(self):
        return self._data


class Autoencoder(nn.Module):
    def __init__(self, input_num_, hidden_num_):
        super(Autoencoder, self).__init__()
        self._input_num = input_num_
        self.fc1 = nn.Linear(input_num_, hidden_num_)
        self.fc2 = nn.Linear(hidden_num_, input_num_)

    def encode(self, x):
        x = func.relu(self.fc1(x))
        return x

    def decode(self, x):
        x = func.relu(self.fc2(x))
        return torch.sigmoid(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def output_representation(self, x):
        return self.encode(x.view(-1, self._input_num))


input_num = 127
hidden_num = 5
log_interval = 1600
batch_size = 32
epoch_num = 30
data_source = 'mimic'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_num, hidden_num).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
file_folder = os.path.abspath('../../resource/preprocessed_data/{}_five_part_five_fold'.format(data_source))
data_loader = DataLoader(file_folder, batch_size)


def main():
    """
    使用变分自编码器学习Patient Feature，注意保留患者顺序
    :return:
    """
    for epoch in range(1, epoch_num + 1):
        train(epoch)
    rep = model.output_representation(torch.from_numpy(data_loader.get_data()).float().to(device))\
        .cpu().data.numpy()
    save_path = os.path.abspath('../../resource/representation/')
    fold_save_size = len(rep) // 5
    np.save(os.path.join(save_path, '{}_pat_representation_{}.npy'
                         .format(data_source, 0)), rep[0: fold_save_size])
    np.save(os.path.join(save_path, '{}_pat_representation_{}.npy'
                         .format(data_source, 1)), rep[fold_save_size: fold_save_size*2])
    np.save(os.path.join(save_path, '{}_pat_representation_{}.npy'
                         .format(data_source, 2)), rep[fold_save_size*2: fold_save_size*3])
    np.save(os.path.join(save_path, '{}_pat_representation_{}.npy'
                         .format(data_source, 3)), rep[fold_save_size*3: fold_save_size*4])
    np.save(os.path.join(save_path, '{}_pat_representation_{}.npy'
                         .format(data_source, 4)), rep[fold_save_size*4:])

    data = data_loader.get_data()
    np.save(os.path.join(save_path, '{}_pat_repre_raw_{}.npy'
                         .format(data_source, 0)), data[0: fold_save_size])
    np.save(os.path.join(save_path, '{}_pat_repre_raw_{}.npy'
                         .format(data_source, 1)), data[fold_save_size: fold_save_size*2])
    np.save(os.path.join(save_path, '{}_pat_repre_raw_{}.npy'
                         .format(data_source, 2)), data[fold_save_size*2: fold_save_size*3])
    np.save(os.path.join(save_path, '{}_pat_repre_raw_{}.npy'
                         .format(data_source, 3)), data[fold_save_size*3: fold_save_size*4])
    np.save(os.path.join(save_path, '{}_pat_repre_raw_{}.npy'
                         .format(data_source, 4)), data[fold_save_size*4:])


def train(epoch_):
    model.train()
    train_loss = 0
    batch_list = data_loader.get_batch_list()
    for batch_idx, data in enumerate(batch_list):
        data = torch.from_numpy(data).float().to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = func.binary_cross_entropy(recon_batch, data, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_, batch_idx * len(data), data_loader.length,
                100. * batch_idx / len(batch_list),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch_, train_loss / data_loader.length))


if __name__ == "__main__":
    main()

