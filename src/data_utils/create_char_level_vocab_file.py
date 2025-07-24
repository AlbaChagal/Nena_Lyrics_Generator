import os

from src.global_constants import clean_lyrics_dir, vocab_file_path


class TextAccumulator:
    """
    A Class that accumulates a list of lyrics into a single text file
    """
    def __init__(self, preprocessed_text_dir: str):
        """
        Initialize
        :param preprocessed_text_dir: path to preprocessed text file
        """
        self.preprocessed_text_dir: str = preprocessed_text_dir
        self.output_path: str = vocab_file_path

    def combine_lyrics(self):
        """
        The main function that combines lyrics into a single text file
        :return: None. Saves the combined lyrics file to self.output_path
        """
        with open(self.output_path, "w", encoding="utf-8") as outfile:
            for fname in os.listdir(self.preprocessed_text_dir):
                if fname.endswith(".txt"):
                    with open(os.path.join(self.preprocessed_text_dir, fname), "r", encoding="utf-8") as infile:
                        outfile.write(infile.read() + "\n\n")


if __name__ == "__main__":
    """
    Create a vocabulary file from lyrics in clean_lyrics_dir
    """
    accumulator = TextAccumulator(preprocessed_text_dir=clean_lyrics_dir)
    accumulator.combine_lyrics()