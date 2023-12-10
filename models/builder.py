# Reference: https://github.com/facebookresearch/moco

import torch
import torch.nn as nn

class DrasCLR(nn.Module):

    def __init__(self, base_encoder, num_patch, rep_dim, moco_dim, num_experts, num_coordinates, K, m, T, mlp):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DrasCLR, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.num_locs = num_patch # add the new dimension of number of locations

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(rep_dim=rep_dim, moco_dim=moco_dim, num_experts=num_experts, num_coordinates=num_coordinates)
        self.encoder_k = base_encoder(rep_dim=rep_dim, moco_dim=moco_dim, num_experts=num_experts, num_coordinates=num_coordinates)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(moco_dim, K, self.num_locs)) # the queue should be the size of (dim of reps) * (number of negative pairs) * (number of total locations)
        self.queue = nn.functional.normalize(self.queue, dim=0) # normalize patch representation
        self.register_buffer("queue_ptr", torch.zeros(self.num_locs, dtype=torch.long)) # set pointer in buffer to 1 for each path location

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, patch_idx):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = self.queue_ptr
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr[patch_idx]:ptr[patch_idx] + batch_size, patch_idx] = keys.T
        ptr[patch_idx] = (ptr[patch_idx] + batch_size) % self.K  # move pointer

        self.queue_ptr = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, patch_idx, pch_q, pch_k, ngb_q):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query patch features
        q, h_q = self.encoder_q(pch_q[0], pch_q[1])  # queries: NxC, encoder needs to take both pathces and their locations as inputs
        q = nn.functional.normalize(q, dim=1)

        # compute query neighbor features
        ngb_flatten = ngb_q[0].reshape(-1, 32, 32, 32)
        loc_flatten = ngb_q[1].reshape(-1, 3)
        r, h_r = self.encoder_q(ngb_flatten[:, None, :, :, :], loc_flatten)
        r = nn.functional.normalize(r, dim=1)
        r = r.reshape(ngb_q[0].shape[0], ngb_q[0].shape[1], -1) # queries: N * R * C, samples * k-neighbors * channels

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            pch_k[0], idx_unshuffle = self._batch_shuffle_ddp(pch_k[0])

            k, h_k = self.encoder_k(pch_k[0], pch_k[1])  # keys: N * C
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # patch InfoNCE logits
        # Einstein sum is more intuitive
        # positive logits: N * 1
        l_pos_pch = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: N * K
        negs = self.queue[:,:,patch_idx].clone().detach() # compute negative logits for each path in the batch conditioned on their locations
        l_neg_pch = torch.einsum('nc,ck->nk', [q, negs])
        # logits: N * (1+K)
        logits_pch = torch.cat([l_pos_pch, l_neg_pch], dim=1)
        # apply temperature
        logits_pch /= self.T

        # neighbor InfoNCE logits
        # positive logits: N * 1
        l_pos_ngb = torch.einsum('nrc, nc->n', [r, k]).unsqueeze(-1)
        # negative logits: N * K
        l_neg_ngb = torch.einsum('nrc, ck->nk', [r, negs])
        # logits: N * (1+K)
        logits_ngb = torch.cat([l_pos_ngb, l_neg_ngb], dim=1)
        # apply temperature
        logits_ngb /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits_pch.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, patch_idx) # consider location for each patch in the batch

        return logits_pch, logits_ngb, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
