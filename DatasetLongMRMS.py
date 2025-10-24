import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class LongMRMSDataset(Dataset):
    """
    Longitudinal dataset for the Long-MR-MS dataset (Ljubljana).
    Returns all studies (timepoints) per patient as a temporal sequence.
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.patient_dirs = sorted(
            [os.path.join(root_dir, p) for p in os.listdir(root_dir) if p.startswith("patient")]
        )

        # build index of patients -> their studies
        self.patients = []
        for patient_dir in self.patient_dirs:
            patient_id = os.path.basename(patient_dir)
            mask_path = glob.glob(os.path.join(patient_dir, f"{patient_id}_gt.nii.gz"))[0]

            studies = sorted(
                list(set(
                    [f.split("_")[1] for f in os.listdir(patient_dir) if "study" in f]
                ))
            )

            study_data = []
            for study in studies:
                flair = glob.glob(os.path.join(patient_dir, f"{patient_id}_{study}_FLAIRreg.nii.gz"))[0]
                t1 = glob.glob(os.path.join(patient_dir, f"{patient_id}_{study}_T1Wreg.nii.gz"))[0]
                t2 = glob.glob(os.path.join(patient_dir, f"{patient_id}_{study}_T2Wreg.nii.gz"))[0]
                study_data.append({
                    "study": study,
                    "flair": flair,
                    "t1": t1,
                    "t2": t2,
                    "mask": mask_path
                })
            self.patients.append({"patient": patient_id, "studies": study_data})

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient["patient"]
        studies = patient["studies"]

        all_images = []
        all_masks = []
        study_ids = []

        def normalize(img):
            nonzero = img > 0
            if np.any(nonzero):
                mean = img[nonzero].mean()
                std = img[nonzero].std()
                img[nonzero] = (img[nonzero] - mean) / (std + 1e-8)
            return img

        for s in studies:
            flair = nib.load(s["flair"]).get_fdata()
            t1 = nib.load(s["t1"]).get_fdata()
            t2 = nib.load(s["t2"]).get_fdata()
            mask = nib.load(s["mask"]).get_fdata()

            # normalize each modality
            flair = normalize(flair)
            t1 = normalize(t1)
            t2 = normalize(t2)

            img = np.stack([flair, t1, t2], axis=0)  # shape (3, H, W, D)
            all_images.append(img)
            all_masks.append(mask[np.newaxis, ...])  # shape (1, H, W, D)
            study_ids.append(s["study"])

        all_images = torch.tensor(np.stack(all_images, axis=0), dtype=torch.float32)  # (T, 3, H, W, D)
        all_masks = torch.tensor(np.stack(all_masks, axis=0), dtype=torch.float32)    # (T, 1, H, W, D)

        if self.transforms:
            all_images, all_masks = self.transforms(all_images, all_masks)

        return {
            "patient": patient_id,
            "studies": study_ids,
            "images": all_images,
            "masks": all_masks
        }

