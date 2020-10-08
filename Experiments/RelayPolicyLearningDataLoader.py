from torch.utils.data import Dataset
import glob
import os
import pickle as pkl


class KitchenDataset(Dataset):
    def __init__(self):
        path = '/home/dhiraj/Downloads/kitchen_demos_multitask_extracted_data'
        self.ctrl = []
        self.qpos = []
        self.qvel = []
        for r in glob.glob(os.path.join(path, '*.pkl')):
            with open(r, 'rb') as f:
                data = pkl.load(f)
                self.ctrl.append(data['ctrl'])
                self.qpos.append(data['qpos'])
                self.qvel.append(data['qvel'])

    def __len__(self):
        return len(self.ctrl)

    def __getitem__(self, item):
        return self.qpos[item], self.ctrl[item]



