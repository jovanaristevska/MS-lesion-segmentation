from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate


class Dataloader(DataLoader):
    """
    Data loading
    """

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=1):
        self.dataset = dataset

        self.shuffle = shuffle

        self.batch_idx = 0

        # if self.shuffle:
        #     self.sampler = RandomSampler(self.dataset)
        # else:
        #     self.sampler = SequentialSampler(self.dataset)
        # self.shuffle = False

        from torch.utils.data import WeightedRandomSampler
        import numpy as np

        if hasattr(self.dataset, "slice_has_lesion"):
            print("⚖️ Using WeightedRandomSampler (lesion oversampling)")
            weights = np.array([1.0 if has else 0.1 for has in self.dataset.slice_has_lesion], dtype=np.float32)
            self.sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        else:
            if self.shuffle:
                self.sampler = RandomSampler(self.dataset)
            else:
                self.sampler = SequentialSampler(self.dataset)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            # 'shuffle': self.shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)


# # data_loader.py
# from torch.utils.data import DataLoader
#
# class Dataloader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
#         super().__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers
#         )
