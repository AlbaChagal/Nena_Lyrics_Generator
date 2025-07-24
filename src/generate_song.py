import os
from typing import Optional, Dict, Any

import torch
from src.global_constants import checkpoints_dir
from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.data_manager import DataManager
from src.train.dataset.word_dataset import WordDataset
from src.train.training_config import TrainingConfig
from train.model import Model
from train.train import Trainer


class SongGenerator:
    """
    An inference class to generate songs from inputting titles to the model
    """
    def __init__(self, model_path: str,
                 training_config: TrainingConfig,
                 data_manager: DataManager):
        """
        Initialize
        :param model_path: path to saved model
        :param training_config: training config
        :param data_manager: data manager
        """
        super(SongGenerator, self).__init__()
        assert os.path.exists(model_path), f'{model_path} does not exist'

        self.training_config: TrainingConfig = training_config
        self.data_manager: DataManager = data_manager
        self.model_path: str = model_path

        self.model: Optional[Model] = None
        self.word2idx: Optional[Dict[str, int]] = None
        self.idx2word: Optional[Dict[int, str]] = None

    def load_model(self, training_config: TrainingConfig):
        """
        Loads model from self.model_path
        :return: None. Saves model, stoi & itos to self
        """
        checkpoint = torch.load(self.model_path, map_location="cpu")
        vocab = checkpoint['vocab']
        if self.training_config.is_use_pretrained_model:
            self.data_manager.embedding_matrix = checkpoint['embedding_matrix']

        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for i, word in enumerate(vocab)}

        # Overwrite vocab size with checkpoint size
        vocab_size = len(vocab)

        # (Optional) Save to data manager so generate() can access it
        self.data_manager.word2idx = self.word2idx
        self.data_manager.idx2word = self.idx2word
        self.data_manager.dataset = None  # prevents accidental mismatch

        # Load embedding_matrix if needed
        if self.training_config.is_use_pretrained_model:
            _, embedding_matrix = self.data_manager.load_data()  # or load from disk
        else:
            embedding_matrix = None

        # Construct model using vocab from checkpoint!
        self.model = Model(
            vocab_size=vocab_size,
            training_config=training_config,
            embedding_matrix=embedding_matrix
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate(self, input_title, length=500, temperature=0.9):
        """
        The main function for generating songs
        :param input_title: title of the song
        :param length: length of the song
        :param temperature: temperature of the song
        :return: The generated song
        """
        assert self.model is not None, 'model must be loaded before calling generate'

        input_title: str = input_title
        if self.training_config.dataset_class == WordDataset:
            tokens = input_title.lower().split()  # or your custom tokenizer
            input_seq = torch.tensor(
                [self.word2idx.get(w, 0) for w in tokens],
                dtype=torch.long
            ).unsqueeze(0)
        elif self.training_config.dataset_class == CharDataset:
            input_seq: torch.Tensor = \
                torch.tensor(
                    [self.data_manager.word2idx.get(c, 0) for c in input_title],
                    dtype=torch.long
                ).unsqueeze(0)
        output_text: str = input_title + '\n\n'

        with torch.no_grad():
            for _ in range(length):
                logits, _ = self.model(input_seq)
                logits = logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_char_idx = torch.multinomial(probs, num_samples=1).item()
                next_char = self.data_manager.idx2word[next_char_idx]
                output_text += next_char

                input_seq = torch.tensor([[next_char_idx]], dtype=torch.long)

        return output_text

if __name__ == "__main__":
    """Infer a model with a single title"""

    title = "101 luftbaloons"
    model_id = '20250724_104255'
    checkpoint_name = 'checkpoint_1001000.pt'

    model_path = os.path.join(checkpoints_dir, model_id, checkpoint_name)

    training_config_main = TrainingConfig()
    data_manager = DataManager(training_config=training_config_main)
    generator = SongGenerator(model_path=model_path, training_config=training_config_main, data_manager=data_manager)
    generator.load_model(training_config_main)
    lyrics = generator.generate(input_title=title)

    print("Generated Lyrics")
    print(lyrics)
