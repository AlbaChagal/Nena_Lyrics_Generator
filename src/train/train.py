import datetime
import json
import os
from typing import Union, Optional, List

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.global_constants import checkpoints_dir, config_json_name
from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.data_manager import DataManager
from src.train.dataset.word_dataset import WordDataset
from src.train.model import Model
from src.train.training_config import TrainingConfig

class Trainer:
    def __init__(self, training_config: TrainingConfig, data_manager: DataManager):
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs_dir: str = os.path.join(checkpoints_dir, self.model_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.training_config: TrainingConfig = training_config
        self.data_manager: DataManager = data_manager
        self.dataset: Optional[Union[CharDataset, WordDataset]] = None

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
            json.dump(training_config.to_string(), f, indent=4)

    def save_checkpoint(self,
                        model_state_dict: dict,
                        optimizer_state_dict: dict,
                        epoch: int,
                        step: int,
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
        y: torch.Tensor
        total_loss: float
        logits: torch.Tensor
        loss: torch.Tensor

        print("Training - start")
        total_num_steps: int = 0
        for epoch in range(self.training_config.num_epochs):
            # print(f'epoch: {epoch + 1} - Start')
            total_loss = 0.0
            for step_in_epoch, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits, _ = model(x)

                loss = F.cross_entropy(logits.view(-1, self.data_manager.dataset.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_num_steps += 1
                if total_num_steps > 0 and total_num_steps % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {total_num_steps}, Loss: {loss.item():.4f}")

                if total_num_steps > 0 and total_num_steps % self.training_config.save_checkpoint_freq == 0:
                    self.save_checkpoint(model_state_dict=model.state_dict(),
                                         optimizer_state_dict=optimizer.state_dict(),
                                         epoch=epoch,
                                         step=total_num_steps,
                                         embedding_matrix=embedding_matrix)



            # print(f"Epoch {epoch + 1} completed, Avg Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    training_config_main = TrainingConfig()
    data_manager = DataManager(training_config_main)
    trainer = Trainer(training_config_main, data_manager=data_manager)
    trainer.train()