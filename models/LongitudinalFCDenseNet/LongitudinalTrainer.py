# import torch
#
# from logger import Mode
# from model.utils.metric import dice_score
# from trainer.Trainer import Trainer
# from utils.illustration_util import log_visualizations_longitudinal
#
#
# class LongitudinalTrainer(Trainer):
#     """
#     Trainer class
#     """
#
#     def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
#                  valid_data_loader=None, lr_scheduler=None, len_epoch=None):
#         super().__init__(model, loss, metric_ftns, optimizer, config, data_loader, valid_data_loader, lr_scheduler, len_epoch)
#
#     def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
#         self.writer.mode = mode#novo
#         _len_epoch = self.len_epoch if mode == Mode.TRAIN else self.len_epoch_val
#
#         # ✅ [1] Log a simple scalar at the start of each epoch
#         if mode == Mode.TRAIN:
#             self.writer.add_scalar("debug/epoch_start", epoch, epoch)
#             self.writer.writer.flush()
#
#         for batch_idx, (x_ref, x, _, target) in enumerate(data_loader):
#             # x_ref, x, target = x_ref.to(self.device), x.to(self.device), target.float().to(self.device), # target.to(self.device)
#
#             x_ref, x = x_ref.to(self.device), x.to(self.device)
#             target = target.float().to(self.device)
#
#             # Log positive pixels
#             if target.shape[1] > 1:
#                 print(f"Batch {batch_idx} | Original target shape: {target.shape}")
#                 print("Target positive pixels per channel:",
#                       [(target[:, c, ...] > 0).sum() for c in range(target.shape[1])])
#
#             # Pick the correct target for loss
#             if self.model.decoder.finalConv.out_channels == 1:
#                 target_for_loss = target[:, 1:2, :, :]  # lesion channel
#             else:
#                 target_for_loss = target # use all channels
#             # Forward pass
#             output = self.model(x_ref, x)
#
#             output_sigmoid = torch.sigmoid(output)
#
#             # ✅ Add this to inspect raw probabilities
#             print(f"Batch {batch_idx} | Raw probabilities (before thresholding):")
#             print("  Min:", output_sigmoid.min().item())
#             print("  Max:", output_sigmoid.max().item())
#             print("  Mean:", output_sigmoid.mean().item())
#             pred_mask = (output_sigmoid > 0.5).float()
#             print("Unique values in pred_mask:", pred_mask.unique())
#             print("Unique values in target_for_loss:", target_for_loss.unique())
#
#             loss = self.loss(output, target_for_loss)
#             dice = dice_score(output, target_for_loss)
#             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Dice: {dice:.4f}")
#
#             # if mode == Mode.TRAIN:
#             #     self.optimizer.zero_grad()
#             # output = self.model(x_ref, x)
#             # loss = self.loss(output, target)
#             if mode == Mode.TRAIN:
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#
#             self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), output, target, loss, mode)
#
#             if not (batch_idx % self.log_step):
#                 self.logger.info(f'{mode.value} Epoch: {epoch} {self._progress(data_loader, batch_idx, _len_epoch)} Loss: {loss.item():.6f}')
#
#             # ✅ [2] Add a test image to confirm visuals appear in TensorBoard
#             if batch_idx == 0 and mode == Mode.TRAIN:
#                 try:
#                     # only first sample in batch, normalize to [0,1] if necessary
#                     img_to_log = (x_ref[0] - x_ref[0].min()) / (x_ref[0].max() - x_ref[0].min() + 1e-8)
#                     self.writer.add_image("debug/x_ref_sample", img_to_log, epoch)
#                     self.writer.writer.flush()
#                 except Exception as e:
#                     self.logger.warning(f"TensorBoard image log failed: {e}")
#
#             if not (batch_idx % (_len_epoch // 10)):
#                 # log_visualizations_longitudinal(self.writer, x_ref, x, output, target, step=self.get_step(batch_idx, epoch, _len_epoch))
#                 log_visualizations_longitudinal(
#                     self.writer, x_ref, x, output, target,
#                     step=self.get_step(batch_idx, epoch, _len_epoch)
#                 )
#             del x_ref, x, target

import torch

from logger import Mode
from model.utils.metric import dice_score
from trainer.Trainer import Trainer
from utils.illustration_util import log_visualizations_longitudinal


class LongitudinalTrainer(Trainer):
    """
    Trainer for longitudinal MR lesion segmentation.
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config,
                         data_loader, valid_data_loader, lr_scheduler, len_epoch)

    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        self.writer.mode = mode
        _len_epoch = self.len_epoch if mode == Mode.TRAIN else self.len_epoch_val

        if mode == Mode.TRAIN:
            self.writer.add_scalar("debug/epoch_start", epoch, epoch)
            self.writer.writer.flush()

        for batch_idx, (x_ref, x, _, target) in enumerate(data_loader):
            # Move tensors to device
            x_ref, x = x_ref.to(self.device), x.to(self.device)
            target = target.float().to(self.device)

            # -------------------------------
            # 1️⃣  Check shapes early
            # -------------------------------
            if batch_idx == 0:
                print(f"[DEBUG] x_ref: {x_ref.shape}, x: {x.shape}, target: {target.shape}")

            # -------------------------------
            # 2️⃣  Forward pass
            # -------------------------------
            output = self.model(x_ref, x)
            output_sigmoid = torch.sigmoid(output)

            # -------------------------------
            # 3️⃣  Adjust target channels
            # -------------------------------
            # If model outputs 1 channel (binary lesion mask)
            if output.shape[1] == 1:
                # Handle case where target has extra dimension
                if target.shape[1] > 1:
                    target_for_loss = target[:, 1:2, :, :]
                else:
                    target_for_loss = target
            else:
                target_for_loss = target

            # -------------------------------
            # 4️⃣  Compute loss and dice
            # -------------------------------
            loss = self.loss(output, target_for_loss)
            dice = dice_score(output_sigmoid, target_for_loss)

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Dice: {dice:.4f}")

            # -------------------------------
            # 5️⃣  Backward + optimizer
            # -------------------------------
            if mode == Mode.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # -------------------------------
            # 6️⃣  Logging
            # -------------------------------
            self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), output, target_for_loss, loss, mode)

            if not (batch_idx % self.log_step):
                self.logger.info(
                    f'{mode.value} Epoch: {epoch} {self._progress(data_loader, batch_idx, _len_epoch)} Loss: {loss.item():.6f}'
                )

            # Visualize samples occasionally
            if batch_idx == 0:
                try:
                    img_to_log = (x_ref[0] - x_ref[0].min()) / (x_ref[0].max() - x_ref[0].min() + 1e-8)
                    self.writer.add_image("debug/x_ref_sample", img_to_log, epoch)
                    self.writer.writer.flush()
                except Exception as e:
                    self.logger.warning(f"TensorBoard image log failed: {e}")

            if not (batch_idx % (_len_epoch // 10)):
                log_visualizations_longitudinal(
                    self.writer, x_ref, x, output, target_for_loss,
                    step=self.get_step(batch_idx, epoch, _len_epoch)
                )

            # Clear memory
            del x_ref, x, target, output, loss
