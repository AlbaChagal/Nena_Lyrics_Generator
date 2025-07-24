from src.data_utils.create_char_level_vocab_file import TextAccumulator
from src.data_utils.preprocess import Preprocessor
from src.data_utils.scrape_lyrics import Scraper
from src.global_constants import raw_lyrics_dir, clean_lyrics_dir


def create_database():
    """
    The complete process of creating a clean data directory from lyrics:
    1. Scrapes lyrics from the internet, using Scraper
    2. Cleans the lyrics, using Preprocessor
    3. Creates vocabulary file, using TextAccumulator
    :return: None
    """
    scraper = Scraper(raw_lyrics_dir)
    scraper.download_all_lyrics()

    preprocessor = Preprocessor(raw_lyrics_dir=raw_lyrics_dir,
                                clean_lyrics_dir=clean_lyrics_dir)
    preprocessor.preprocess_lyrics()

    accumulator = TextAccumulator(preprocessed_text_dir=clean_lyrics_dir)
    accumulator.combine_lyrics()


if __name__ == "__main__":
    """ Runs entire database creation process """
    create_database()