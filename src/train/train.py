import datetime
import json
import os
from typing import Union, Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.global_constants import checkpoints_dir, config_json_name, tensorboard_log_dir, eos_token, new_line_token

from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.data_manager import DataManager
from src.train.dataset.title_to_lyrics_dataset import TitleToLyricsDataset
from src.train.dataset.word_dataset import WordDataset
from src.train.losses.losses_datastructs import AllLosses
from src.train.losses.regularization_loss import WordsInTextCounter
from src.train.losses.training_losses import LossFunction


from src.train.metrics.metrics_calculator import AllMetricsCalculator
from src.train.metrics.metrics_datastructs import AllMetrics
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
        self.device: torch.device = self.get_device()
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs_dir: str = os.path.join(checkpoints_dir, self.model_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.training_config: TrainingConfig = training_config
        self.data_manager: DataManager = data_manager
        self.dataset: Optional[Union[CharDataset, WordDataset, TitleToLyricsDataset]] = None
        self.train_tensorboard_logger: TensorboardLogger = \
            TensorboardLogger(os.path.join(self.outputs_dir, tensorboard_log_dir, 'train'))
        self.val_tensorboard_logger: TensorboardLogger = \
            TensorboardLogger(os.path.join(self.outputs_dir, tensorboard_log_dir, 'val'))
        self.loss_function: LossFunction = LossFunction(self.training_config, device=self.device)
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

    def train(self):

        print(f'started training model: {self.model_id} on device: {self.device}')

        # Load data and split into train/val
        dataloader: DataLoader = self.data_manager.load_data()
        word2idx: Dict[str, int] = self.data_manager.dataset.word2idx
        # Split ttl_dataloader into train and val
        ttl_dataset: TitleToLyricsDataset = dataloader.dataset
        val_split = self.training_config.percentage_of_data_to_use_as_validation
        val_size = int(len(ttl_dataset) * val_split)
        train_size = len(ttl_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(ttl_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Validation dataset size: {len(val_dataset)}')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        all_metrics_calculator: AllMetricsCalculator = AllMetricsCalculator(word2idx, device=self.device)
        if self.is_debug:
            print('Initialized model')

        vocab_size = len(self.data_manager.dataset.vocab)
        model: Model = Model(vocab_size=vocab_size,
                             training_config=self.training_config).to(self.device)
        if self.is_debug:
            print(f'Initialized model with vocab_size={vocab_size}')

        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(),
                                                            lr=self.training_config.learning_rate)
        
        if self.is_debug:
            print('Initialized Optimizer')

        self.save_training_config()

        total_losses: AllLosses = AllLosses()
        logits: torch.Tensor
        loss: torch.Tensor

        title: torch.Tensor

        print("Training - start")
        total_num_steps: int = 0


        for epoch in range(self.training_config.num_epochs):
            for ttl_step_in_epoch, (input_tensor, target_tensor) in enumerate(train_loader):
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                max_length = target_tensor.size(1)
                logits = model(src=input_tensor, tgt=target_tensor, max_length=max_length, start_token_idx=0, return_logits=True)
                assert self.data_manager.dataset is not None, "Dataset is not initialized"

                # Always slice target to match logits' sequence length for loss computation
                seq_len_logits = logits.shape[1]
                y_target = target_tensor[:, 1:1+seq_len_logits]
                # Compute all losses (main + special tokens)
                loss, step_all_losses = self.loss_function(logits=logits, labels=y_target, word2idx=word2idx)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_losses += step_all_losses

                # Compute metrics for all special tokens
                y_target_flat = y_target[0].tolist()
                all_metrics: AllMetrics = all_metrics_calculator.calculate(logits=logits, labels=y_target)

                with torch.no_grad():
                    idx2char = getattr(self.data_manager.dataset, 'idx2word', None)
                    if idx2char is None:
                        idx2char = {}
                    word_percentage_in_output: float = self.word_regulator_loss.count_valid_words(
                        logits=logits,
                        idx2char=idx2char
                    )

                # Calculate output length (tokens until <EOS> in target)
                eos_token_id = word2idx[eos_token]
                if eos_token_id in y_target_flat:
                    output_length = y_target_flat.index(eos_token_id) + 1
                else:
                    output_length = len(y_target_flat)
                self.train_tensorboard_logger.update(
                    all_losses=total_losses,
                    all_metrics=all_metrics,
                    word_percentage_in_output=word_percentage_in_output,
                    output_length=output_length
                )

                total_num_steps += 1
                if total_num_steps > 0 and total_num_steps % 100 == 0:
                    total_losses /= 100
                    print(f"Epoch {epoch + 1}, Step {total_num_steps}, {total_losses}")
                    self.train_tensorboard_logger.log(total_num_steps)

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
                            val_input = val_input.to(self.device)
                            val_target = val_target.to(self.device)
                            max_length = val_target.size(1)
                            val_logits = model(src=val_input, tgt=val_target, max_length=max_length, start_token_idx=0, return_logits=True)
                            # Always slice target to match logits' sequence length for loss computation
                            seq_len_logits = val_logits.shape[1]
                            y_target = val_target[:, 1:1+seq_len_logits]
                            # Compute all losses (main + special tokens)
                            val_loss_val, val_all_losses = self.loss_function(logits=val_logits, labels=y_target, word2idx=word2idx)
                            val_loss += val_loss_val.item()
                            # Compute metrics for all special tokens
                            y_target_flat = y_target[0].tolist()
                            val_all_metrics: AllMetrics = all_metrics_calculator.calculate(logits=val_logits, labels=y_target)
                            idx2char = getattr(self.data_manager.dataset, 'idx2word', None)
                            if idx2char is None:
                                idx2char = {}
                            word_percentage_in_output_val: float = self.word_regulator_loss.count_valid_words(
                                logits=val_logits,
                                idx2char=idx2char
                            )
                            eos_token_id = word2idx[eos_token]
                            if eos_token_id in y_target_flat:
                                output_length = y_target_flat.index(eos_token_id) + 1
                            else:
                                output_length = len(y_target_flat)
                            self.val_tensorboard_logger.update(
                                all_losses=val_all_losses,
                                all_metrics=val_all_metrics,
                                word_percentage_in_output=word_percentage_in_output_val,
                                output_length=output_length
                            )
                    avg_val_loss = val_loss / len(val_loader)
                    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        self.train_tensorboard_logger.close()


if __name__ == "__main__":
    is_debug_main = False

    training_config_main = TrainingConfig()
    random_state = np.random.RandomState(training_config_main.random_seed)
    data_manager = DataManager(training_config_main, random_state=random_state, is_debug=is_debug_main)
    trainer = Trainer(training_config_main, data_manager=data_manager, is_debug=is_debug_main)
    trainer.train()