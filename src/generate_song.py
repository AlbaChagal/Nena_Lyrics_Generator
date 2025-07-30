import os
from typing import Optional, Dict, Any

import numpy as np
import torch
from src.global_constants import checkpoints_dir
from src.train.dataset.data_manager import DataManager
from src.train.training_config import TrainingConfig, DatasetType
from src.train.model import Model


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

        # Construct model using vocab from checkpoint!
        self.model = Model(
            vocab_size=vocab_size,
            training_config=training_config
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate(self, input_title: str, length: int = 500, temperature: float = 0.7):
        """
        Autoregressive generation: feed title and all generated words so far, stop at <EOS> or max length.
        """
        assert self.model is not None, 'model must be loaded before calling generate'
        assert self.word2idx is not None, "word2idx must be set before generating lyrics"
        assert self.idx2word is not None, "idx2word must be set before generating lyrics"

        # Tokenize input title
        if self.training_config.dataset_class == DatasetType.WordDataset:
            # Use the tokenizer from data_manager for consistency
            assert hasattr(self.data_manager, "tokenizer") and self.data_manager.tokenizer is not None, "Tokenizer must be set in data_manager."
            tokens = self.data_manager.tokenizer.tokenize(input_title)
            input_ids = [self.word2idx.get(tok, self.word2idx.get("<UNK>")) for tok in tokens]
        elif self.training_config.dataset_class == DatasetType.CharDataset:
            input_ids = [self.word2idx.get(c, self.word2idx.get("<UNK>", 0)) for c in input_title]
        else:
            raise ValueError("Unsupported dataset class for generation")

        # Start with input_ids (title tokens)
        generated_ids = input_ids.copy()
        eos_token_id = self.word2idx.get("<EOS>")
        unk_token_id = self.word2idx.get("<UNK>")

        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                input_tensor = torch.tensor([generated_ids], dtype=torch.long)
                # Get logits for the last token
                logits = self.model(src=input_tensor, return_last_logits=True)  # (1, vocab_size)
                next_token_logits = logits[0] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                generated_ids.append(next_token_id)
                if next_token_id == eos_token_id:
                    break

        # Convert generated_ids to tokens, skip title tokens in output
        # Filter out None values (should not happen, but for safety)
        output_tokens = [self.idx2word[idx] for idx in generated_ids[len(input_ids):]]
        # Stop at <EOS> if present
        if "<EOS>" in output_tokens:
            output_tokens = output_tokens[:output_tokens.index("<EOS>")]

        # Join output appropriately
        if self.training_config.dataset_class == DatasetType.WordDataset:
            output_text = " ".join(output_tokens)
        else:
            output_text = "".join(output_tokens)

        return output_text.strip()

if __name__ == "__main__":
    """Infer a model with a single title"""

    title = "101 luftbaloons"
    model_id = '20250730_110916'
    checkpoint_name = 'checkpoint_1016000.pt'
    model_dir = os.path.join(checkpoints_dir, model_id)
    model_path = os.path.join(model_dir, checkpoint_name)

    # training_config_main = TrainingConfig.from_json(os.path.join(model_dir, config_json_name))
    training_config_main = TrainingConfig()
    random_state = np.random.RandomState(training_config_main.random_seed)
    data_manager = DataManager(training_config=training_config_main,
                               random_state=random_state)
    generator = SongGenerator(model_path=model_path,
                              training_config=training_config_main,
                              data_manager=data_manager)
    generator.load_model(training_config_main)
    lyrics = generator.generate(input_title=title)

    print("Generated Lyrics")
    print(lyrics)
