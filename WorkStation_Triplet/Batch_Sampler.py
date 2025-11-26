import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes):
        self.labels = torch.as_tensor(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes
        assert batch_size % num_classes == 0
        self.n_per_class = batch_size // num_classes

        self.class_indices = []
        for c in range(num_classes):
            idx = torch.nonzero(self.labels == c, as_tuple=False).view(-1).tolist()
            self.class_indices.append(idx)

        self.max_len = max(len(idx_list) for idx_list in self.class_indices)

    def __iter__(self):
        ptr = [0 for _ in range(self.num_classes)]
        perm = [torch.randperm(len(idx_list)).tolist() for idx_list in self.class_indices]

        num_batches = (self.max_len * self.num_classes) // self.batch_size
        for _ in range(num_batches):
            batch = []
            for c in range(self.num_classes):
                idx_list = self.class_indices[c]
                p = ptr[c]
                order = perm[c]
                for _ in range(self.n_per_class):
                    if p >= len(order):
                        order = torch.randperm(len(idx_list)).tolist()
                        p = 0
                    batch.append(idx_list[order[p]])
                    p += 1
                ptr[c] = p
                perm[c] = order
            yield batch

    def __len__(self):
        return (self.max_len * self.num_classes) // self.batch_size
