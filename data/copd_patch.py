from torch.utils.data import Dataset
import numpy as np
import glob
from sklearn.metrics import pairwise_distances

DATA_DIR = "/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/"

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, args, patch_transforms=default_transform, neighbor_transforms=default_transform):
        self.stage = stage
        self.args = args
        self.root_dir = args.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.patch_transforms = patch_transforms
        self.neighbor_transforms = neighbor_transforms

        # atlas patch locations, our refernce file can be found at ./preprocess/misc/atlas_patch_loc.npy
        self.patch_loc = np.load(self.args.root_dir + "19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        # pairwise distance
        self.dists = pairwise_distances(self.patch_loc, metric='euclidean')
        # normalize patch locations
        self.patch_loc = (self.patch_loc / self.patch_loc.max(0)) * 2 - 1  # normalize position to [-1, 1]

        self.patch_idx = 0
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")
        # top k nearest patches
        self.k_neighbor_idx = np.argsort(self.dists[self.patch_idx,:])[1: (self.args.k_neighbors+1)]
        neighbor_lst = []
        for k in range(self.args.k_neighbors):
            neighbor_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.k_neighbor_idx[k])+".npy")
            neighbor_lst.append(neighbor_data[None, :, :, :, :]) # 1 * 9179 * 32 * 32 * 32
        self.neighbor_data = np.concatenate(neighbor_lst, axis=0)
        del neighbor_lst

        if stage == 'training':
            # Specific to COPDGene dataset, you can change depends on your needs
            FILE = open(DATA_DIR + "phase1_Final_10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.label_name]
            race_idx = mylist.index("race")
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                tmp = [mylist[idx] for idx in metric_idx]
                if "" in tmp:
                    continue
                if self.args.nhw_only and mylist[race_idx] != "1":
                    continue
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]] = metric_list
            FILE.close()

        if stage == 'testing':
            # Specific to COPDGene dataset, you can change depends on your needs
            self.label_name = self.args.label_name + self.args.label_name_set2
            FILE = open(DATA_DIR + "phase1_Final_10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.label_name]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                tmp = [mylist[idx] for idx in metric_idx]
                if "" in tmp[:3]:
                    continue
                metric_list = []
                for i in range(len(metric_idx)):
                    if tmp[i] == "":
                        metric_list.append(-1024)
                    else:
                        metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]] = metric_list + [-1024, -1024, -1024]
            FILE = open(DATA_DIR + "CT_scan_datasets/CT_visual_scoring/COPDGene_CT_Visual_20JUL17.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.visual_score]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                if mylist[0] not in self.metric_dict:
                    continue
                tmp = [mylist[idx] for idx in metric_idx]
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]][
                -len(self.args.visual_score) - len(self.args.P2_Pheno):-len(self.args.P2_Pheno)] = metric_list
            FILE.close()
            FILE = open(
                DATA_DIR + 'P1-P2 First 5K Long Data/Subject-flattened- one row per subject/First5000_P1P2_Pheno_Flat24sep16.txt',
                'r')
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.P2_Pheno]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                if mylist[0] not in self.metric_dict:
                    continue
                tmp = [mylist[idx] for idx in metric_idx]
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]][-len(self.args.P2_Pheno):] = metric_list
            FILE.close()

        self.sid_list = []
        for item in glob.glob(self.args.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        assert len(self.sid_list) == self.patch_data.shape[0]

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_patch_idx(self, patch_idx):
        self.patch_idx = patch_idx
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")
        # top k nearest patches
        self.k_neighbor_idx = np.argsort(self.dists[self.patch_idx,:])[1: (self.args.k_neighbors+1)]
        neighbor_lst = []
        for k in range(self.args.k_neighbors):
            neighbor_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.k_neighbor_idx[k])+".npy")
            neighbor_lst.append(neighbor_data[None, :, :, :, :]) # 1 * 9179 * 32 * 32 * 32
        self.neighbor_data = np.concatenate(neighbor_lst, axis=0)
        del neighbor_lst

    def __len__(self):
        if self.stage == 'training':
            return self.sid_list_len * self.args.num_patch
        if self.stage == 'testing':
            return self.sid_list_len

    def __getitem__(self, idx):

        if self.stage == 'training':
            idx = idx % self.sid_list_len

            # patch data
            pch = self.patch_data[idx, :, :, :]
            pch = np.clip(pch, -1024, 240) # clip input intensity to [-1024, 240]
            pch = pch + 1024.
            pch = self.patch_transforms(pch[None, :, :, :])
            pch[0] = pch[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
            pch[1] = pch[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
            # patch location
            patch_loc_idx = self.patch_loc[self.patch_idx, :]

            # neighbor data
            ngb = self.neighbor_data[:, idx, :, :, :]
            ngb = np.clip(ngb, -1024, 240) # clip input intensity to [-1024, 240]
            ngb = ngb + 1024.
            ngb = self.neighbor_transforms(ngb)
            ngb = ngb/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
            # neighbor location
            neighor_loc_idx = self.patch_loc[self.k_neighbor_idx, :]

            # labels
            key = self.sid_list[idx][:6]
            label = np.asarray(self.metric_dict[key])
            return key, pch, patch_loc_idx, ngb, neighor_loc_idx, label

        if self.stage == 'testing':
            sid = self.sid_list[idx]

            # read the entire image including 581 patches
            img = np.load(self.root_dir + "patch/" + sid + "_patch.npy")
            img = np.clip(img, -1024, 240)  # clip input intensity to [-1024, 240]
            img = img + 1024.
            img = img[:, None, :, :, :] / 632. - 1  # Normalize to [-1,1], 632=(1024+240)/2

            # patch locations for all 581 patches
            patch_loc_idx = self.patch_loc

            # study id
            key = self.sid_list[idx][:6]

            # labels
            label = np.asarray(self.metric_dict[key])  # extract sid from the first 6 letters

            return sid, img, patch_loc_idx, label