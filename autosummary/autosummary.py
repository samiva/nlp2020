
"""
This is the main module of the autosummary project. Currently it offers both
the CLI and the actual automatic summarization functionality.
"""

import argparse
import logging
import re
import string
import urllib.request

from typing import Sequence, Optional

import matplotlib.pyplot
import nltk.tokenize

from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.corpus import stopwords


_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
_STOP_WORDS = stopwords.words('english')


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_type",
                        type=str,
                        choices=("url", "file"),
                        help="Specifies the type of the source_path: URL or filepath.")
    parser.add_argument("source_path",
                        type=str,
                        help="Filepath or URL pointing to source.")
    return parser


def _chapter_from_html(html: str, chapter_header: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for header in soup.find_all(re.compile('^h[1-6]$')):
        if header.get_text().upper() == chapter_header.upper():
            # Get the text of the following item after the chapter header
            return header.next_sibling.get_text()


def _get_raw_source(type_: str, path: str) -> str:
    """Handle the source fetching: get the raw HTML out of filepath or url"""
    def _read_file(filepath: str) -> str:
        with open(filepath, "r") as f:
            contents = f.read()
        return contents

    def _read_url(url: str) -> str:
        return urllib.request.urlopen(url).read()

    # Specifies the handler functions for different source types
    source_handler = {
        "file": _read_file,
        "url": _read_url,
    }
    return source_handler[type_](path)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOGGING_FORMAT)
    _parsed_args = _argument_parser().parse_args()
    source_type = _parsed_args.source_type
    source_path = _parsed_args.source_path
    _logger.debug("Using source '{}' ({})".format(source_path, source_type.upper()))

    raw_html = _get_raw_source(source_type, source_path)
    # Get the titles & headers
    titles = _titles_from_html(raw_html)
    _logger.debug("TITLES: {}".format(titles))
    # Get the abstract (if it exists)
    abstract = _chapter_from_html(raw_html, "abstract")
    _logger.debug("ABSTRACT: {}".format(abstract))

    if abstract is None:
        _logger.info("No abstract found.")
        return

    # Preprocessing
    abstract = abstract.lower()
    abstract = _remove_punctuation(abstract)
    tokens = _tokenize_text(abstract, nltk.tokenize.word_tokenize)
    tokens_without_stopwords = _remove_stopwords(tokens, _STOP_WORDS)
    stemmed_tokens = _stemming(tokens_without_stopwords, nltk.PorterStemmer())

    _logger.debug("TOKENS: {}".format(stemmed_tokens))

    # Frequency distribution
    frequency_distribution = _word_frequency_distribution(stemmed_tokens)
    _plot_frequency_distribution(frequency_distribution)


def _plot_frequency_distribution(fdist: FreqDist) -> None:
    """TODO: Generate histogram"""
    fdist.plot()
    #matplotlib.pyplot.hist(fdist.tabulate())


def _remove_punctuation(text: str) -> str:
    return "".join([c for c in text if c not in string.punctuation])


def _remove_stopwords(tokens: Sequence[str], wordlist: Sequence[str]) -> Sequence[str]:
    return [w for w in tokens if w not in wordlist]


def _stemming(tokens: Sequence[str], stemmer) -> Sequence[str]:
    return [stemmer.stem(w) for w in tokens]


def _titles_from_html(html: str) -> Sequence[str]:
    """Returns the title(s) and headers as a list."""
    soup = BeautifulSoup(html, "html.parser")
    # Get the title(s)
    titles = [a.get_text() for a in soup.find_all("title")]
    # Get all headers in the order of appearance
    # https://stackoverflow.com/questions/45062534/how-to-grab-all-headers-from-a-website-using-beautifulsoup
    titles.extend(a.get_text().strip("¶") for a in soup.find_all(re.compile('^h[1-6]$')))
    return titles


def _tokenize_text(text: str, tokenizer) -> Sequence[str]:
    return tokenizer(text)


def _word_frequency_distribution(tokens: Sequence[str]) -> FreqDist:
    return FreqDist(tokens)


if __name__ == "__main__":
    main()
