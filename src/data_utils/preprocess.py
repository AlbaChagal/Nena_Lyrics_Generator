import os
import re

from src.global_constants import raw_lyrics_dir as raw_lyrics_dir_main
from src.global_constants import clean_lyrics_dir as clean_lyrics_dir_main

class Preprocessor:
    """
    A Class for preprocessing the raw lyrics dataset
    """
    def __init__(self, raw_lyrics_dir: str, clean_lyrics_dir: str):
        """
        Initialize
        :param raw_lyrics_dir: path to raw lyrics dir
        :param clean_lyrics_dir: path to clean lyrics dir
        """
        self.raw_lyrics_dir: str = raw_lyrics_dir
        self.clean_lyrics_dir: str = clean_lyrics_dir

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a single text
        :param text: lyrics text
        :return: cleaned lyrics text
        """
        text: str = re.sub(r"\[.*?\]", "", text)  # remove annotations
        text = re.sub(r"\n+", "\n", text)    # normalize newlines
        text.replace(r"Kein Songtext vorhanden", " ")
        return text.strip()

    @staticmethod
    def clean_title(title: str) -> str:
        """
        Clean a single title
        :param title: lyrics title
        :return: cleaned lyrics title
        """
        title: str = re.sub(r"_+", "_", title)  # normalize underscores
        return title

    def preprocess_lyrics(self):
        """
        The entire process of preprocessing
        :return: None. Saves preprocessed lyrics files to
                       self.clean_lyrics_dir with clean_titles as file names
        """
        os.makedirs(self.clean_lyrics_dir, exist_ok=True)
        for filename in os.listdir(self.raw_lyrics_dir):
            with open(os.path.join(self.raw_lyrics_dir, filename), "r", encoding="utf-8") as infile:
                raw: str = infile.read()
            cleaned: str = self.clean_text(raw)
            cleaned_title: str = self.clean_title(filename)
            with open(os.path.join(self.clean_lyrics_dir, cleaned_title), "w", encoding="utf-8") as outfile:
                outfile.write(cleaned)

if __name__ == "__main__":
    """
    Create a clean data directory from lyrics in raw_lyrics_dir
    """
    preprocessor = Preprocessor(raw_lyrics_dir=raw_lyrics_dir_main,
                                clean_lyrics_dir=clean_lyrics_dir_main)
    preprocessor.preprocess_lyrics()
