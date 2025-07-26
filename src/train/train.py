import datetime
import json
import os
from typing import Union, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.global_constants import checkpoints_dir, config_json_name, tensorboard_log_dir

from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.data_manager import DataManager
from src.train.dataset.word_dataset import WordDataset
from src.train.losses.regularization_loss import WordRegulator
from src.train.losses.training_losses import NextCharacterLoss, MaskedCharacterLoss
from src.train.model import Model
from src.train.tensorboard_logger import TensorboardLogger
from src.train.training_config import TrainingConfig

class Trainer:
    def __init__(self, training_config: TrainingConfig, data_manager: DataManager):
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs_dir: str = os.path.join(checkpoints_dir, self.model_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.training_config: TrainingConfig = training_config
        self.data_manager: DataManager = data_manager
        self.dataset: Optional[Union[CharDataset, WordDataset]] = None
        self.tensorboard_logger: TensorboardLogger = \
            TensorboardLogger(os.path.join(self.outputs_dir, tensorboard_log_dir))

        self.next_loss: NextCharacterLoss = NextCharacterLoss()
        self.mask_loss: MaskedCharacterLoss = MaskedCharacterLoss()
        self.word_regulator_loss: WordRegulator = WordRegulator()

    @staticmethod
    def get_device() -> torch.device:
        device: torch.device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def get_checkpoint_name(self, epoch: int, step: int) -> str:
        return f"{self.outputs_dir}/checkpoint_{step + int((epoch + 1) * 1000000)}.pt"

    def save_training_config(self) -> None:
        """
        Saves the training configuration from self to json file
        :return: None
        """
        training_config: TrainingConfig = self.training_config
        with open(os.path.join(self.outputs_dir, config_json_name), "w") as f:
            json.dump(training_config.serialize(), f, indent=4)

    def save_checkpoint(self,
                        model_state_dict: dict,
                        optimizer_state_dict: dict,
                        epoch: int,
                        step: int,
                        total_num_steps: int,
                        embedding_matrix) -> None:
        checkpoint_path: str = self.get_checkpoint_name(epoch, step)
        vocab: Union[List[int], List[str]] = self.data_manager.vocab
        torch.save(
            obj={
                'model_state_dict': model_state_dict,
                'vocab':  vocab,
                'optimizer': optimizer_state_dict,
                'embedding_matrix': embedding_matrix
            },
            f=checkpoint_path
        )
        print(f"Model saved to {checkpoint_path}")
        self.tensorboard_logger.log(total_num_steps)

    def calc_all_losses(self,
                        step: int,
                        logits: torch.Tensor,
                        x: torch.tensor,
                        y_next: torch.Tensor,
                        y_original: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        loss_next: torch.Tensor = self.next_loss(logits=logits,
                                   y=y_next,
                                   vocab_size=self.data_manager.dataset.vocab_size)

        loss_masked: torch.Tensor = self.mask_loss.forward(x=x,
                                             y=y_original,
                                             logits=logits,
                                             mask_token_idx=self.data_manager.dataset.mask_token_idx)

        loss_word_regulation: torch.Tensor = self.word_regulator_loss.forward(
            logits=logits,
            idx2char=self.data_manager.dataset.idx2word
        )

        weighted_loss_next: torch.Tensor = loss_next * (1 - self.training_config.mask_loss_weight)
        weighted_loss_masked: torch.Tensor = loss_masked * self.training_config.mask_loss_weight

        word_reg_weight: float = self.training_config.word_regularization_loss_weight if step > 10000 else 0.
        weighted_loss_word_regulation: torch.Tensor = loss_word_regulation * word_reg_weight

        loss: torch.Tensor = weighted_loss_next + weighted_loss_masked + weighted_loss_word_regulation

        return loss, weighted_loss_next, weighted_loss_masked, weighted_loss_word_regulation


    def train(self):
        device: torch.device = self.get_device()
        print(f'started training model: {self.model_id} on device: {device}')

        dataloader: DataLoader
        embedding_matrix: Optional[torch.Tensor]
        print('Loading data managers - Start')
        dataloader, embedding_matrix = self.data_manager.load_data()
        print('Loading data managers - Finish')

        print('Initializing model - Start')
        model: Model = Model(vocab_size=self.data_manager.dataset.vocab_size,
                             embedding_matrix=embedding_matrix,
                             training_config=self.training_config).to(device)
        print('Initializing model - Finish')

        print('Initializing Optimizer - Start')
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(),
                                                            lr=self.training_config.learning_rate)
        print('Initializing Optimizer - Finish')

        self.save_training_config()

        checkpoint_path: str
        x: torch.Tensor
        y_next: torch.Tensor
        y_original: torch.Tensor
        total_loss: float
        logits: torch.Tensor
        loss_next: torch.Tensor
        weighted_loss_next: torch.Tensor
        loss_masked: torch.Tensor
        weighted_loss_masked: torch.Tensor
        loss_word_regulation: torch.Tensor
        weighted_loss_word_regulation: torch.Tensor
        loss: torch.Tensor

        print("Training - start")
        total_num_steps: int = 0
        for epoch in range(self.training_config.num_epochs):
            total_loss = 0.0
            for step_in_epoch, (x, y_next, y_original) in enumerate(dataloader):
                x = x.to(device)
                y_next = y_next.to(device)
                y_original = y_original.to(device)

                optimizer.zero_grad()
                logits, _ = model(x)

                loss, weighted_loss_next, weighted_loss_masked, weighted_loss_word_regulation = \
                    self.calc_all_losses(
                        logits=logits,
                        x=x,
                        y_next=y_next,
                        y_original=y_original,
                        step=total_num_steps
                    )

                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                total_loss += loss_item
                self.tensorboard_logger.update(loss=loss_item,
                                               next_loss=weighted_loss_next.item(),
                                               masked_loss=weighted_loss_masked.item(),
                                               word_regularization_loss=weighted_loss_word_regulation.item())
                total_num_steps += 1
                if total_num_steps > 0 and total_num_steps % 100 == 0:
                    print(f"Epoch {epoch + 1}, "
                          f"Step {total_num_steps}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Next Character Loss: {weighted_loss_next.item(): .4f}, "
                          f"Masked Character Loss: {weighted_loss_masked.item(): .4f}, "
                          f"Word Regularization Loss: {weighted_loss_word_regulation.item(): .4f}")

                if total_num_steps > 0 and total_num_steps % self.training_config.save_checkpoint_freq == 0:
                    self.save_checkpoint(model_state_dict=model.state_dict(),
                                         optimizer_state_dict=optimizer.state_dict(),
                                         epoch=epoch,
                                         step=total_num_steps,
                                         total_num_steps=total_num_steps,
                                         embedding_matrix=embedding_matrix)
        self.tensorboard_logger.close()


if __name__ == "__main__":
    training_config_main = TrainingConfig()
    random_state = np.random.RandomState(training_config_main.random_seed)
    data_manager = DataManager(training_config_main, random_state=random_state)
    trainer = Trainer(training_config_main, data_manager=data_manager)
    trainer.train()