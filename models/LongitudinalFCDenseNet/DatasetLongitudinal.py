# import os
#
# import h5py
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
#
# from dataset.dataset_utils import Phase, Modalities, Mode, retrieve_data_dir_paths, Evaluate
#
#
# class DatasetLongitudinal(Dataset):
#     """DatasetLongitudinal dataset"""
#
#     def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patients=None, evaluate: Evaluate = Evaluate.TRAINING, preprocess=True, view=None):
#         self.modalities = list(map(lambda x: Modalities(x), modalities))
#         self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patients, Mode.LONGITUDINAL, view)
#
#         self.data_dir_paths = [p for p in self.data_dir_paths if 'patient1' in p[0]]
#     def __len__(self):
#         return len(self.data_dir_paths)
#
#     def __getitem__(self, idx):
#         x_ref, x, ref_label, label = [], [], None, None
#         x_ref_path, x_path = self.data_dir_paths[idx]
#         for i, modality in enumerate(self.modalities):
#             with h5py.File(os.path.join(x_ref_path, f'{modality.value}.h5'), 'r') as f:
#                 x_ref.append(f['data'][()])
#                 if ref_label is None:
#                     ref_label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)
#
#             with h5py.File(os.path.join(x_path, f'{modality.value}.h5'), 'r') as f:
#                 x.append(f['data'][()])
#                 if label is None:
#                     label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)
#         return torch.as_tensor(x_ref).float(), torch.as_tensor(x).float(), ref_label.float(), label.float()
import math
import os
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

import numpy as np
from dataset.dataset_utils import Phase, Modalities, Mode, retrieve_data_dir_paths, Evaluate


class DatasetLongitudinal(Dataset):
    """DatasetLongitudinal dataset with optional slice limiting and resized slices for model input."""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(),
                 val_patients=None, evaluate: Evaluate = Evaluate.TRAINING,
                 preprocess=True, view='AXIAL', patch_size=(256, 256), max_slices=None, target_size = (217, 217)): #target_size = (217, 217) vaka bese
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.data_dir_paths = retrieve_data_dir_paths(
            data_dir, evaluate, phase, preprocess, val_patients, Mode.LONGITUDINAL, view
        )

        # 2️⃣ Filter out empty slices (image or label all zeros)
        filtered_paths = []
        for t0_slice_path, t1_slice_path in self.data_dir_paths:
            # You can open one of them (e.g., t0) to check data and label
            h5_file = os.path.join(t0_slice_path, 'FLAIR.h5')  # or any modality you use
            if not os.path.exists(h5_file):
                continue
            with h5py.File(h5_file, "r") as f:
                img = f["data"][:]
                label = f["label"][:]

                # 🧠 Add this line to inspect what you are filtering
                print("Example slice:", img.shape, "Label sum:", np.sum(label))
                print("Label unique values:", np.unique(label))

                if np.sum(img) > 1e-5 and np.sum(label) > 1e-5:
                    filtered_paths.append((t0_slice_path, t1_slice_path))

        self.data_dir_paths = filtered_paths
        print(f"✅ Using {len(self.data_dir_paths)} valid slices after filtering")

        # Optional slice limit
        if max_slices is not None:
            self.data_dir_paths = self.data_dir_paths[:max_slices]

        self.patch_size = patch_size
        self.phase = phase

        # ✅ Identify which slices contain lesions for weighted sampling
        self.slice_has_lesion = []
        for t0_slice_path, _ in self.data_dir_paths:
            h5_file = os.path.join(t0_slice_path, 'FLAIR.h5')
            with h5py.File(h5_file, "r") as f:
                label = f["label"][:]
                has_lesion = np.sum(label) > 0
                self.slice_has_lesion.append(has_lesion)

        lesion_count = sum(self.slice_has_lesion)
        print(
            f"🧩 Lesion slices: {lesion_count}/{len(self.slice_has_lesion)} ({lesion_count / len(self.slice_has_lesion):.1%})")

    def __len__(self):
        return len(self.data_dir_paths)

    def _random_crop_same(self, img_ref, ref_label, img, label):
        _, h, w = img_ref.shape
        ph, pw = self.patch_size
        if h <= ph or w <= pw:
            return img_ref, ref_label, img, label

        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)

        img_ref_patch = img_ref[:, top:top + ph, left:left + pw]
        ref_label_patch = ref_label[:, top:top + ph, left:left + pw]
        img_patch = img[:, top:top + ph, left:left + pw]
        label_patch = label[:, top:top + ph, left:left + pw]

        return img_ref_patch, ref_label_patch, img_patch, label_patch


    def _augment(self, img, label):
        """Lightweight augmentations: flip, rotate, intensity scale."""
        # Random horizontal/vertical flips
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            label = torch.flip(label, dims=[2])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
            label = torch.flip(label, dims=[1])

        # Small rotation (±10 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            angle_rad = math.radians(angle)
            theta = torch.tensor([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0]
            ], dtype=torch.float)
            grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size(), align_corners=False)
            img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', align_corners=False).squeeze(0)
            label = F.grid_sample(label.unsqueeze(0), grid, mode='nearest', align_corners=False).squeeze(0)

        # Brightness / intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            img = torch.clamp(img * scale, 0, 1)

        return img, label

    # def __getitem__(self, idx): vaka beshe
    #     x_ref, x, ref_label, label = [], [], None, None
    #     x_ref_path, x_path = self.data_dir_paths[idx]
    #
    #     for modality in self.modalities:
    #         # Load reference image
    #         with h5py.File(os.path.join(x_ref_path, f'{modality.value}.h5'), 'r') as f:
    #             img_ref = torch.as_tensor(f['data'][()], dtype=torch.float32).unsqueeze(0)  # [1,H,W]
    #             img_ref = F.interpolate(img_ref.unsqueeze(0), size=self.target_size,
    #                                     mode='bilinear', align_corners=False).squeeze(0)
    #             x_ref.append(img_ref)
    #
    #             if ref_label is None:
    #                 lbl_ref = torch.as_tensor(f['label'][()], dtype=torch.int64)  # [H,W]
    #                 ref_label = F.one_hot(lbl_ref, num_classes=2).permute(2, 0, 1).float()  # [C,H,W]
    #                 ref_label = F.interpolate(ref_label.unsqueeze(0), size=self.target_size,
    #                                           mode='nearest').squeeze(0)
    #
    #         # Load current image
    #         with h5py.File(os.path.join(x_path, f'{modality.value}.h5'), 'r') as f:
    #             img = torch.as_tensor(f['data'][()], dtype=torch.float32).unsqueeze(0)
    #             img = F.interpolate(img.unsqueeze(0), size=self.target_size,
    #                                 mode='bilinear', align_corners=False).squeeze(0)
    #             x.append(img)
    #
    #             if label is None:
    #                 lbl = torch.as_tensor(f['label'][()], dtype=torch.int64)  # [H,W]
    #                 label = F.one_hot(lbl, num_classes=2).permute(2, 0, 1).float()
    #                 label = F.interpolate(label.unsqueeze(0), size=self.target_size,
    #                                       mode='nearest').squeeze(0)
    #
    #     # Concatenate modalities into channel dimension
    #     x_ref = torch.cat(x_ref, dim=0)  # [C*num_modalities, H, W]
    #     x = torch.cat(x, dim=0)          # [C*num_modalities, H, W]
    #
    #     return x_ref, x, ref_label, label

    def __getitem__(self, idx):
        x_ref, x, ref_label, label = [], [], None, None
        x_ref_path, x_path = self.data_dir_paths[idx]

        for modality in self.modalities:
            # Load reference
            with h5py.File(os.path.join(x_ref_path, f'{modality.value}.h5'), 'r') as f:
                img_ref = torch.as_tensor(f['data'][()], dtype=torch.float32).unsqueeze(0)
                lbl_ref = torch.as_tensor(f['label'][()], dtype=torch.int64)
            ref_label = F.one_hot(lbl_ref, num_classes=2).permute(2, 0, 1).float()

            # Load current
            with h5py.File(os.path.join(x_path, f'{modality.value}.h5'), 'r') as f:
                img = torch.as_tensor(f['data'][()], dtype=torch.float32).unsqueeze(0)
                lbl = torch.as_tensor(f['label'][()], dtype=torch.int64)
            label = F.one_hot(lbl, num_classes=2).permute(2, 0, 1).float()

            x_ref.append(img_ref)
            x.append(img)

        x_ref = torch.cat(x_ref, dim=0)
        x = torch.cat(x, dim=0)

        # Random patch cropping (same region for ref & current)
        # x_ref, ref_label = self._random_crop(x_ref, ref_label)
        # x, label = self._random_crop(x, label)
        x_ref, ref_label, x, label = self._random_crop_same(x_ref, ref_label, x, label)

        # Apply augmentations only during training
        if self.phase == Phase.TRAIN:
            x_ref, ref_label = self._augment(x_ref, ref_label)
            x, label = self._augment(x, label)

        return x_ref, x, ref_label, label




# from torch.utils.data import Dataset
# import torch
# import nibabel as nib
# import numpy as np
# from dataset.dataset_utils import retrieve_data_dir_paths, Modalities
# from enum import Enum
#
# class Phase(Enum):
#     TRAIN = 'train'
#     VAL = 'val'
#     TEST = 'test'
#
# class DatasetLongitudinal(Dataset):
#     def __init__(self, data_dir, phase=Phase.TRAIN, modalities=("flair","t1","t2")):
#         self.phase = phase
#         self.modalities = modalities
#         self.data_entries = retrieve_data_dir_paths(data_dir, phase)
#
#     def __len__(self):
#         return len(self.data_entries)
#
#     def __getitem__(self, idx):
#         entry = self.data_entries[idx]
#         imgs = []
#         for mod in self.modalities:
#             img = nib.load(entry["images"][mod]).get_fdata()
#             imgs.append(img)
#         img = np.stack(imgs, axis=0)  # shape: (modalities, H, W, D)
#         mask = nib.load(entry["mask"]).get_fdata()
#
#         # convert to torch tensors
#         img = torch.as_tensor(img, dtype=torch.float32)
#         mask = torch.as_tensor(mask, dtype=torch.float32)
#
#         return img, mask
