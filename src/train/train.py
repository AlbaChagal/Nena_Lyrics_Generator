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
from src.train.losses.regularization_loss import WordsInTextCounter
from src.train.losses.training_losses import NextCharacterLoss, MaskedCharacterLoss, TitleToLyricsLoss
from src.train.model import Model
from src.train.tensorboard_logger import TensorboardLogger
from src.train.training_config import TrainingConfig

class Trainer:
    def __init__(self, training_config: TrainingConfig, data_manager: DataManager, is_debug: bool = False):
        """
        Initializes the Trainer with the given training configuration and data manager.
        :param training_config: Training configuration object containing hyperparameters and settings.
        :param data_manager: Data manager object to handle datasets and tokenization.
        :param is_debug: If True, enables debug mode for additional logging.
        """

        self.is_debug: bool = is_debug
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs_dir: str = os.path.join(checkpoints_dir, self.model_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.training_config: TrainingConfig = training_config
        self.data_manager: DataManager = data_manager
        self.dataset: Optional[Union[CharDataset, WordDataset]] = None
        self.train_tensorboard_logger: TensorboardLogger = \
            TensorboardLogger(os.path.join(self.outputs_dir, tensorboard_log_dir, 'train'))
        self.val_tensorboard_logger: TensorboardLogger = \
            TensorboardLogger(os.path.join(self.outputs_dir, tensorboard_log_dir, 'val'))

        self.next_loss: NextCharacterLoss = NextCharacterLoss()
        self.mask_loss: MaskedCharacterLoss = MaskedCharacterLoss()
        self.ttl_loss: TitleToLyricsLoss = TitleToLyricsLoss()
        self.word_regulator_loss: WordsInTextCounter = WordsInTextCounter(is_debug=self.is_debug)

    @staticmethod
    def get_device(is_debug: bool = False) -> torch.device:
        device: torch.device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        if is_debug:
            print(f"Using device: {device}")
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
        if self.is_debug:
            print(f"Training configuration saved to {os.path.join(self.outputs_dir, config_json_name)}")

    def save_checkpoint(self,
                        model_state_dict: dict,
                        optimizer_state_dict: dict,
                        epoch: int,
                        step: int,
                        total_num_steps: int,
                        embedding_matrix: Optional[torch.Tensor]) -> None:
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

        self.train_tensorboard_logger.log(total_num_steps)
        print(f"logged training step {total_num_steps} to Tensorboard at {self.train_tensorboard_logger.log_dir}")

    def calc_loss(self, logits: torch.Tensor, y: torch.Tensor, vocab_size: int, is_debug: bool = False) -> torch.Tensor:
        # logits: (batch_size, seq_len, vocab_size)
        # y: (batch_size, seq_len)
        if is_debug:
            print(f"logits.shape: {logits.shape}")
            print(f"y.shape: {y.shape}")
            print(f"vocab_size: {vocab_size}")
            print(f"logits.numel(): {logits.numel()}, y.numel(): {y.numel()}")
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 is <PAD>
        seq_len_logits = logits.shape[1]
        y_target = y[:, 1:1+seq_len_logits]  # match logits' sequence length
        logits_flat = logits.contiguous().view(-1, vocab_size)
        y_flat = y_target.contiguous().view(-1)
        if is_debug:
            print(f"logits_flat.shape: {logits_flat.shape}, y_flat.shape: {y_flat.shape}")
        loss: torch.Tensor = loss_fn(logits_flat, y_flat)
        return loss * self.training_config.loss_weight


    def train(self):

        device: torch.device = self.get_device()
        print(f'started training model: {self.model_id} on device: {device}')

        # Load data and split into train/val
        dataloader = self.data_manager.load_data()
        # Split ttl_dataloader into train and val
        ttl_dataset = dataloader.dataset
        val_split = 0.1
        val_size = int(len(ttl_dataset) * val_split)
        train_size = len(ttl_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(ttl_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Validation dataset size: {len(val_dataset)}')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        assert self.data_manager.dataset is not None, "Dataset is not initialized"
        model: Model = Model(vocab_size=len(self.data_manager.dataset),
                             training_config=self.training_config).to(device)
        if self.is_debug:
            print('Initialized model')

        vocab_size = len(self.data_manager.dataset.vocab)
        model: Model = Model(vocab_size=vocab_size,
                            training_config=self.training_config).to(device)
        if self.is_debug:
            print(f'Initialized model with vocab_size={vocab_size}')

        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(),
                                                            lr=self.training_config.learning_rate)
        
        if self.is_debug:
            print('Initialized Optimizer')

        self.save_training_config()

        total_loss: float
        logits: torch.Tensor
        loss: torch.Tensor

        title: torch.Tensor

        print("Training - start")
        total_num_steps: int = 0

        for epoch in range(self.training_config.num_epochs):
            total_loss = 0.0
            # The dataset now yields (input, target) pairs for random prefixes (autoregressive training)
            for ttl_step_in_epoch, (input_tensor, target_tensor) in enumerate(train_loader):
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
                max_length = target_tensor.size(1)
                logits = model(src=input_tensor, tgt=target_tensor, max_length=max_length, start_token_idx=0, return_logits=True)
                assert self.data_manager.dataset is not None, "Dataset is not initialized"
                vocab_size = len(self.data_manager.dataset.vocab)
                loss = self.calc_loss(logits=logits, y=target_tensor, vocab_size=vocab_size, is_debug=self.is_debug)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item = loss.item()
                total_loss += loss_item

                with torch.no_grad():
                    idx2char = getattr(self.data_manager.dataset, 'idx2word', None)
                    if idx2char is None:
                        idx2char = {}
                    word_percentage_in_output: float = self.word_regulator_loss.count_valid_words(
                        logits=logits,
                        idx2char=idx2char
                    )

                self.train_tensorboard_logger.update(loss=loss_item, 
                                                     word_percentage_in_output=word_percentage_in_output)
                total_num_steps += 1
                if total_num_steps > 0 and total_num_steps % 100 == 0:
                    print(f"Epoch {epoch + 1}, "
                          f"Step {total_num_steps}, "
                          f"Loss: {loss.item():.4f}, " 
                          f"Word Percentage in Output: {word_percentage_in_output:.4f}")

                if total_num_steps > 0 and total_num_steps % self.training_config.save_checkpoint_freq == 0:
                    self.save_checkpoint(model_state_dict=model.state_dict(),
                                         optimizer_state_dict=optimizer.state_dict(),
                                         epoch=epoch,
                                         step=total_num_steps,
                                         total_num_steps=total_num_steps,
                                         embedding_matrix=None)

                    val_loss = 0.0
                    with torch.no_grad():
                        for val_input, val_target in val_loader:
                            val_input = val_input.to(device)
                            val_target = val_target.to(device)
                            max_length = val_target.size(1)
                            val_logits = model(src=val_input, tgt=val_target, max_length=max_length, start_token_idx=0, return_logits=True)
                            vocab_size = len(self.data_manager.dataset.vocab)
                            loss = self.calc_loss(logits=val_logits, y=val_target, vocab_size=vocab_size)
                            val_loss += loss.item()

                            idx2char = getattr(self.data_manager.dataset, 'idx2word', None)
                            if idx2char is None:
                                idx2char = {}
                            word_percentage_in_output_val: float = self.word_regulator_loss.count_valid_words(
                                logits=val_logits,
                                idx2char=idx2char
                            )

                            self.val_tensorboard_logger.update(loss=loss.item(), 
                                                               word_percentage_in_output=word_percentage_in_output_val)
                    avg_val_loss = val_loss / len(val_loader)
                    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
                    # Log validation loss and perplexity to Tensorboard
                    self.val_tensorboard_logger.log(total_num_steps)

        self.train_tensorboard_logger.close()


if __name__ == "__main__":
    is_debug_main = False

    training_config_main = TrainingConfig()
    random_state = np.random.RandomState(training_config_main.random_seed)
    data_manager = DataManager(training_config_main, random_state=random_state, is_debug=is_debug_main)
    trainer = Trainer(training_config_main, data_manager=data_manager, is_debug=is_debug_main)
    trainer.train()