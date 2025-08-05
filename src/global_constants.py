import os
from typing import Callable, List

"""Constants used throughout the project"""

project_dir = "/Users/shaharheyman/PycharmProjects/nena_lyrics_generator"

# Global strings
no_lyrics_on_site_str = 'Kein Songtext vorhanden.'
mask_character = '¤'

# Data directories
data_dir = os.path.join(project_dir, "data")
raw_lyrics_dir = os.path.join(data_dir, "raw_lyrics")
clean_lyrics_dir = os.path.join(data_dir, "clean_lyrics")
char_vocab_file_path = os.path.join(data_dir, "char_all_lyrics.txt")
word_vocab_file_path = os.path.join(data_dir, "word_all_lyrics.txt")

# Training directories
checkpoints_dir = os.path.join(project_dir, "checkpoints")
config_json_name = "config.json"
tensorboard_log_dir = "tensorboard_logs"

# Special Tokens
eos_token: str = "<EOS>"
new_line_token: str = "\n"
separator_token: str = "<SEP>"
bos_token: str = "<BOS>"
unknown_token: str = "<UNK>"
def get_all_special_tokens() -> List[str]:
    return [eos_token, new_line_token, separator_token, bos_token, unknown_token]