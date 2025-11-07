import argparse
import os
from collections import defaultdict
from copy import copy

import numpy as np
import torch

import data_loader as module_data_loader
import dataset as module_dataset
import model as module_arch
import model.utils.loss as module_loss
import model.utils.metric as module_metric
import trainer as trainer_module
from dataset.DatasetStatic import Phase
from dataset.dataset_utils import Views
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume=None):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if resume:
        config.resume = resume

    logger = config.get_logger('train')

    # get function handles of loss and metrics
    # loss_class = getattr(module_loss, config['loss'])
    # loss = loss_class()
    loss_class = getattr(module_loss, config['loss']['type'])
    loss = loss_class(**config['loss']['args'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # setup data_loader instances
    if config['single_view']:
        results = defaultdict(list)
        for view in list(Views):
            _cfg = copy(config)
            logs = train(logger, _cfg, loss, metrics, view=view)
            for k, v in list(logs.items()):
                results[k].append(v)

    else:
        train(logger, config, loss, metrics)


def train(logger, config, loss, metrics, view: Views = None):
    # dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.TRAIN, view=view)
    dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.TRAIN)
    data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'], dataset=dataset)

    # val_dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.VAL, view=view)
    val_dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.VAL)
    valid_data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'], dataset=val_dataset)

    # build model architecture, then print to console
    model = config.initialize_class('arch', module_arch)
    logger.info(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if view:
        config._save_dir = os.path.join(config._save_dir, str(view.name))
        config._log_dir = os.path.join(config._log_dir, str(view.name))
        os.mkdir(config._save_dir)
        os.mkdir(config._log_dir)
    trainer = config.retrieve_class('trainer', trainer_module)(model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler)
    return trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--single_view', default=False, type=bool, help='Defines if a single is used per plane orientation')

    config = ConfigParser(*parse_cmd_args(args))
    main(config)





# import argparse
# import os
# from collections import defaultdict
# from copy import copy
# import numpy as np
# import torch
#
# # Use your simplified dataset
# from dataset.MyDataset import MyDataset
# import dataset as module_dataset
# import data_loader as module_data_loader
# import model as module_arch
# import model.utils.loss as module_loss
# import model.utils.metric as module_metric
# import trainer as trainer_module
# from parse_config import ConfigParser, parse_cmd_args
#
#
# def main(config, resume=None):
#     torch.manual_seed(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(0)
#
#     if resume:
#         config.resume = resume
#
#     logger = config.get_logger('train')
#
#     # get function handles of loss and metrics
#     loss = getattr(module_loss, config['loss'])
#     metrics = [getattr(module_metric, met) for met in config['metrics']]
#
#     # Setup datasets
#     train_dataset = config.retrieve_class('dataset', module_dataset)(
#         **config['dataset']['args'], phase=Phase.TRAIN
#     )
#     val_dataset = config.retrieve_class('dataset', module_dataset)(
#         **config['dataset']['args'], phase=Phase.VAL
#     )
#
#     # Setup dataloaders
#     train_loader = config.retrieve_class('data_loader', module_data_loader)(
#         **config['data_loader']['args'], dataset=train_dataset
#     )
#     val_loader = config.retrieve_class('data_loader', module_data_loader)(
#         **config['data_loader']['args'], dataset=val_dataset
#     )
#
#     # Build model
#     model = config.initialize_class('arch', module_arch)
#     logger.info(model)
#
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = config.initialize('optimizer', torch.optim, trainable_params)
#     lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
#
#     # Initialize trainer
#     trainer = config.retrieve_class(
#         'trainer', trainer_module
#     )(model, loss, metrics, optimizer, config, train_loader, val_loader, lr_scheduler)
#
#     trainer.train()
#
#
# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser(description='Train LongitudinalFCDenseNet')
#     # parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
#     # parser.add_argument('-r', '--resume', default=None, type=str, help='checkpoint path')
#     # parser.add_argument('-d', '--device', default=None, type=str, help='GPU device(s)')
#     # args = parser.parse_args()
#
#     # from parse_config import ConfigParser, parse_cmd_args
#     # config = ConfigParser(*parse_cmd_args())
#     # main(config)
#
#     import argparse
#     from parse_config import ConfigParser, parse_cmd_args
#
#     # 1️⃣ Create parser
#     parser = argparse.ArgumentParser(description='Train LongitudinalFCDenseNet')
#     parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
#     parser.add_argument('-r', '--resume', default=None, type=str, help='checkpoint path')
#     parser.add_argument('-d', '--device', default=None, type=str, help='GPU device(s)')
#     parser.add_argument('-s', '--single_view', default=False, type=bool, help='Use single plane view')
#
#     # 2️⃣ Pass parser to parse_cmd_args
#     parsed_args = parse_cmd_args(parser)
#
#     # 3️⃣ Create config
#     config = ConfigParser(*parsed_args)
#
#     # 4️⃣ Run main
#     main(config)




# import argparse
# import numpy as np
# import torch
# from parse_config import ConfigParser, parse_cmd_args
#
# # Import your custom dataset and data loader
# from dataset.MyDataset import MyDataset
# from data_loader import Dataloader  # replace with your actual DataLoader class
# import model as module_arch
# import model.utils.loss as module_loss
# import model.utils.metric as module_metric
# import trainer as trainer_module
#
# def main(config, resume=None):
#     torch.manual_seed(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(0)
#
#     if resume:
#         config.resume = resume
#
#     logger = config.get_logger('train')
#
#     # get function handles of loss and metrics
#     loss = getattr(module_loss, config['loss'])
#     metrics = [getattr(module_metric, met) for met in config['metrics']]
#
#     # Setup datasets with phase
#     train_dataset = MyDataset(**config['dataset']['args'], phase='train')
#     val_dataset = MyDataset(**config['dataset']['args'], phase='val')
#
#     # Setup dataloaders
#     # On Windows, set num_workers=0 to avoid pickling issues
#     data_loader_args = config['data_loader']['args'].copy()
#     data_loader_args['num_workers'] = 0
#
#     train_loader = Dataloader(**data_loader_args, dataset=train_dataset)
#     val_loader = Dataloader(**data_loader_args, dataset=val_dataset)
#
#     # Build model
#     model = config.initialize_class('arch', module_arch)
#     logger.info(model)
#
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = config.initialize('optimizer', torch.optim, trainable_params)
#     lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
#
#     # Initialize trainer
#     trainer = config.retrieve_class(
#         'trainer', trainer_module
#     )(model, loss, metrics, optimizer, config, train_loader, val_loader, lr_scheduler)
#
#     trainer.train()
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train LongitudinalFCDenseNet')
#     parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
#     parser.add_argument('-r', '--resume', default=None, type=str, help='checkpoint path')
#
#     # First, parse the command-line args
#     args = parser.parse_args()
#
#     # Pass parsed args to parse_cmd_args
#     config_dict, resume = parse_cmd_args(args)
#
#     # Now initialize ConfigParser
#     config = ConfigParser(config_dict, resume=resume)
#
#     # Call main
#     main(config, resume=resume)
#



