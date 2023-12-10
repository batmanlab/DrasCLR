import os
import argparse
import json
from easydict import EasyDict as edict
from tqdm import tqdm

import random
import numpy as np
import torch

from models.cnn3d import Encoder

from data.copd_patch import COPD_dataset

parser = argparse.ArgumentParser(description='Extract 3D Images Representations')
parser.add_argument('--exp-name', default='./ssl_exp/exp_neighbor_0_128')
parser.add_argument('--checkpoint-patch', default='checkpoint_patch_0001.pth.tar')
parser.add_argument('--batch-size', type=int, default=1)


def main():
    # read configurations
    p = parser.parse_args()
    patch_epoch = p.checkpoint_patch.split('.')[0][-4:]
    with open(os.path.join(p.exp_name, 'configs.json')) as f:
        args = edict(json.load(f))
    args.checkpoint = os.path.join(p.exp_name, p.checkpoint_patch)
    args.batch_size = p.batch_size
    args.patch_rep_dir = os.path.join(p.exp_name, 'patch_rep', patch_epoch)
    os.makedirs(args.patch_rep_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    main_worker(args)

def main_worker(args):
    #args.gpu = 0
    #torch.cuda.set_device(args.gpu)

    # create patch-level encoder
    model_patch = Encoder(rep_dim=args.rep_dim_patch, moco_dim=args.moco_dim_patch, num_experts=args.num_experts, num_coordinates=args.num_coordinates)

    # remove the last FC layer
    model_patch.fc = torch.nn.Sequential()
    state_dict = torch.load(args.checkpoint)['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model_patch.load_state_dict(state_dict)
    print(model_patch)
    print("Patch model weights loaded.")
    #model_patch.cuda()
    model_patch = torch.nn.DataParallel(model_patch).cuda()
    model_patch.eval()

    # dataset
    test_dataset_patch = COPD_dataset('testing', args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset_patch, batch_size=1, shuffle=False,
        num_workers=10, pin_memory=True, drop_last=False)
    args.label_name = args.label_name + args.label_name_set2

    # train dataset
    sid_lst = []
    pred_arr = np.empty((len(test_dataset_patch), args.num_patch, args.rep_dim_patch))
    feature_arr = np.empty(
        (len(test_dataset_patch), len(args.label_name) + len(args.visual_score) + len(args.P2_Pheno)))
    iterator = tqdm(test_loader,
                    desc="Propagating (X / X Steps)",
                    bar_format="{r_bar}",
                    dynamic_ncols=True,
                    disable=False)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            sid, images, patch_loc_idx, labels = batch
            sid_lst.append(sid[0])
            images = images[0].float().cuda()
            patch_loc_idx = patch_loc_idx[0].float().cuda()
            _, pred = model_patch(images, patch_loc_idx)
            pred_arr[i, :, :] = pred.cpu().numpy()
            feature_arr[i:i+1, :] = labels
            iterator.set_description("Propagating (%d / %d Steps)" % (i, len(test_dataset_patch)))
    np.save(os.path.join(args.patch_rep_dir, "sid_arr_full.npy"), sid_lst)
    np.save(os.path.join(args.patch_rep_dir, "pred_arr_patch_full.npy"), pred_arr)
    np.save(os.path.join(args.patch_rep_dir, "feature_arr_patch_full.npy"), feature_arr)
    print("\nExtraction patch representation on full set finished.")


if __name__ == '__main__':
    main()

