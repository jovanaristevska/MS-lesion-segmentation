import os

CONFIG = {
    "name": f"{os.path.basename(__file__).split('.')[0]}",
    "n_gpu": 0,

    "arch": {
        "type": "LongitudinalFCDenseNet",
        "args": {
            # "in_channels": 3, vaka bese
            "in_channels": 1,
            "siamese": True
        }
    },
    "dataset": {
        "type": "DatasetLongitudinal",
        "args": {
            "data_dir": "D:/manu_project/data/long-MR-MS_preprocessed",
            "preprocess": False,
            # "modalities":  ["FLAIR", "T1W", "T2W"],
            "modalities":  ["FLAIR"],
            "val_patients": [0, 1, 2, 3,],
            "max_slices": 300, #novo,
            "view" : 'AXIAL'

        }
    },
    "data_loader": {
        "type": "Dataloader",
        "args": {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.001,
            # "weight_decay": 0, vaka bese
            "weight_decay": 1e-5,
            "amsgrad": True
        }
    },
    # "loss": "BCEDiceLoss",
    "loss": {
        "type": "WeightedBCEDiceLoss",
        "args": {
            "pos_weight": 30.0,   # try 10 first, then 20 if still all zeros
            "bce_weight": 0.5
        }
    },
    # "loss": "FocalDiceLoss",
    "metrics": [
        "precision", "recall", "dice_loss", "dice_score", "asymmetric_loss"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 5,
            "gamma": 0.5
        }
    },
    "trainer": {
        "type": "LongitudinalTrainer",
        "epochs": 3,
        "save_dir": "../saved3ep",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "max val_dice_score",
        "early_stop": 2,
        "tensorboard": True
    }
}

# import os
#
# CONFIG = {
#     "name": f"{os.path.basename(__file__).split('.')[0]}",
#     "n_gpu": 0,
#
#     "arch": {
#         "type": "LongitudinalFCDenseNet",
#         "args": {
#             "in_channels": 3,
#             "siamese": False
#         }
#     },
#     "dataset": {
#         "type": "MyDataset",
#         "args": {
#             "data_dir": "D:/manu_project/data/long-MR-MS_patient01-05",
#             "modalities": ["flair", "t1", "t2"],
#             "transform": None
#         }
#     },
#     "data_loader": {
#         "type": "Dataloader",
#         "args": {
#             "batch_size": 2,
#             "shuffle": True,
#             "num_workers": 2,
#         }
#     },
#     "optimizer": {
#         "type": "Adam",
#         "args": {
#             "lr": 0.0001,
#             "weight_decay": 0,
#             "amsgrad": True
#         }
#     },
#     "loss": "mse",
#     "metrics": [
#         "precision", "recall", "dice_loss", "dice_score", "asymmetric_loss"
#     ],
#     "lr_scheduler": {
#         "type": "StepLR",
#         "args": {
#             "step_size": 50,
#             "gamma": 0.1
#         }
#     },
#     "trainer": {
#         "type": "LongitudinalTrainer",
#         "epochs": 100,
#         "save_dir": "../saved/",
#         "save_period": 1,
#         "verbosity": 2,
#         "monitor": "min val_dice_loss",
#         "early_stop": 10,
#         "tensorboard": True
#     }
# }
#
