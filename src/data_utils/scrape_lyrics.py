from bs4 import BeautifulSoup, element
import os
import re
import requests
import tqdm
from typing import Dict

from src.global_constants import no_lyrics_on_site_str, raw_lyrics_dir


class Scraper(object):
    """
    A Class for scraping lyrics of the web - hard-coded set to scraping Nena lyrics from songtexte.com
    """
    def __init__(self, output_dir: str):
        """
        Initialize
        :param output_dir: path to output dir
        """
        self.nena_url_suffix: str = "artist/nena-4bd6ff92.html"
        self.lyrics_url_prefix: str = "songtext/nena/"
        self.base_url: str = "https://www.songtexte.com/"
        self.output_dir: str = output_dir

    @staticmethod
    def clean_song_name_from_html(song_name: str) -> str:
        """
        Removes unwanted characters from song name
        :param song_name: The name of the song to clean
        :return: cleaned song name
        """
        return song_name.split('Nena - ')[-1].split(' Songtext')[0]

    @staticmethod
    def scrape_lyrics(song_url: str):
        """
        Scrape lyrics from a URL (This will probably not work for any website other than - songtexte.com)
        :param song_url: The URL of the song to scrape
        :return: scraped lyrics
        """
        response = requests.get(song_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        lyric_box = soup.find("div", {"id": "lyrics"})
        if lyric_box:
            for br in lyric_box.find_all("br"):
                br.replace_with("\n")
            lyrics = lyric_box.get_text(separator="\n").strip()
            lyrics = re.sub(r"\n+", "\n", lyrics)
            return lyrics
        else:
            print(f'No lyrics found for {song_url}')
            return None

    def get_url_from_href(self, href: str) -> str:
        """
        Retrieve the url from a href
        :param href: The href string
        :return: The url
        """
        clean_href = href.split('../')[-1]
        return self.base_url + clean_href

    def get_next_page_url(self, bs: BeautifulSoup) -> str:
        """
        Load the next songs page
        :param bs: The BeautifulSoup object
        :return: next page url
        """
        result: element.ResultSet = \
            bs.find_all(name='a',
                        title=re.compile('Go to next page'),
                        href=re.compile('../artist/nena'))
        assert len(result) <= 1, \
            f'there should only be 1 next page button ' \
            f'or none on last page, got {len(result)}'

        return self.get_url_from_href(result[0]['href'])

    def get_song_links_from_single_page(self, bs: BeautifulSoup) -> Dict[str, str]:
        """
        Get all song-links from a single page
        :param bs: The BeautifulSoup object
        :return: A dictionary with song names as keys and song links as values
        """
        song_links: element.ResultSet = (
            bs.find_all(name='a',
                        href=re.compile(f'../{self.lyrics_url_prefix}')))
        song_links_dict: Dict[str, str] = \
            {self.clean_song_name_from_html(link['title']):
                 self.get_url_from_href(link['href']) for link in song_links}
        return song_links_dict

    def get_all_song_links(self) -> Dict[str, str]:
        """
        Get all song-links from all pages
        :return: A dictionary with song names as keys and song links as values
        """
        song_links_dict: Dict[str, str] = {}
        url: str = self.base_url + self.nena_url_suffix
        while True:
            response: requests.models.Response = requests.get(url)
            bs: BeautifulSoup = BeautifulSoup(response.content, 'html.parser')
            song_links_dict.update(self.get_song_links_from_single_page(bs))
            try:
                url = self.get_next_page_url(bs)
            except IndexError:
                break

        return song_links_dict

    def save_lyrics_for_single_song(self, title: str, lyrics: str):
        """
        Save lyrics for a single song
        :param title: The title of the song
        :param lyrics: The lyrics of the song
        :return: None. Saves the lyrics.txt file to self.output_dir/{safe_title}.txt
        """
        os.makedirs(self.output_dir, exist_ok=True)
        safe_title = re.sub(r"[^\w\-_.]+", "_", title)
        with open(f"{self.output_dir}/{safe_title}.txt", "w") as f:
            f.write(lyrics)

    def download_all_lyrics(self):
        """
        The entire process of downloading all raw lyrics
        :return: None. Saves the raw lyrics to data/raw_lyrics_dir
        """
        songs_dict = self.get_all_song_links()
        lyrics_found = 0
        for title, lyrics_link in tqdm.tqdm(songs_dict.items(), desc="Scraping songs"):
            lyrics = self.scrape_lyrics(lyrics_link)
            if lyrics:
                if no_lyrics_on_site_str in lyrics:
                    continue
                self.save_lyrics_for_single_song(title=title, lyrics=lyrics)
                lyrics_found += 1
        print(f"Found {lyrics_found} songs out of {len(songs_dict)}. Saved to: {self.output_dir}")


if __name__ == "__main__":
    """
    Scrape all lyrics from all Nena songs
    """
    output_dir_main = raw_lyrics_dir
    scraper = Scraper(output_dir_main)
    scraper.download_all_lyrics()