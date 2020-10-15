
"""
This is the main module of the autosummary project. Currently it offers both
the CLI and the actual automatic summarization functionality.
"""

import argparse
import logging
import re
import string
import urllib.request

from typing import Dict, Sequence, Optional

import matplotlib.pyplot
import nltk.tokenize
import spacy

from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.corpus import stopwords


_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
_NAMED_ENTITY_TAGS = ("PERSON", "ORG")
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
    parser.add_argument("-w",
                        dest="word_count",
                        type=int,
                        default=10,
                        help="Number of high frequency words to include for the summarization.")
    return parser


def _chapter_from_html(html: str, chapter_header: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for header in soup.find_all(re.compile('^h[1-6]$')):
        if header.get_text().upper() == chapter_header.upper():
            # Get the text of the following item after the chapter header
            #_logger.debug("{}: {}".format(header.get_text().upper(), type(header.next_sibling)))
            try:
                return header.next_sibling.get_text()
            except AttributeError:
                # The element object didn't have the get_text() attribute meaning
                # it's not likely a text chapter anyway.
                return None


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


def _headers_from_html(html: str) -> Sequence[str]:
    """Returns the headers as a list of strings."""
    soup = BeautifulSoup(html, "html.parser")
    # Get the title(s)
    titles = [a.get_text() for a in soup.find_all("title")]
    # Get all headers in the order of appearance
    # https://stackoverflow.com/questions/45062534/how-to-grab-all-headers-from-a-website-using-beautifulsoup
    titles.extend(a.get_text().strip("¶") for a in soup.find_all(re.compile('^h[1-6]$')))
    return titles


def _highest_freq_words(freq_dist: FreqDist, word_count: int) -> Sequence[str]:
    return [a[0] for a in freq_dist.most_common(word_count)]


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOGGING_FORMAT)
    _parsed_args = _argument_parser().parse_args()
    source_type = _parsed_args.source_type
    source_path = _parsed_args.source_path
    word_count = _parsed_args.word_count
    _logger.debug("Using source '{}' ({})".format(source_path, source_type.upper()))

    raw_html = _get_raw_source(source_type, source_path)
    # Get the titles & headers
    titles = _titles_from_html(raw_html)
    headers = _headers_from_html(raw_html)
    _logger.debug("HEADERS: {}".format(headers))
    texts_by_chapters = _texts_by_chapters(raw_html, headers)
    _logger.debug(texts_by_chapters)

    text = _text_from_html(raw_html)
    if text is None:
        _logger.info("No text found.")
        return

    # Preprocessing
    text = text.lower()
    text_processed = _remove_punctuation(text)
    tokens = _tokenize_text(text_processed, nltk.tokenize.word_tokenize)
    tokens_without_stopwords = _remove_stopwords(tokens, _STOP_WORDS)
    stemmed_tokens = _stemming(tokens_without_stopwords, nltk.PorterStemmer())
    _logger.debug("TOKENS: {}".format(stemmed_tokens))

    # Frequency distribution
    freq_dist = _word_frequency_distribution(tokens_without_stopwords)
    _plot_frequency_distribution(freq_dist)

    # TODO: Needed for determining whether a sentence is part of a specific chapter
    sentences_by_chapters = _sentences_by_chapters(texts_by_chapters)

    named_ents = _named_entities_from_text(text, _NAMED_ENTITY_TAGS)
    _logger.info("NAMED ENTITIES: {}".format(named_ents))
    high_freq_words = _highest_freq_words(freq_dist, word_count)
    _logger.info("HIGH FREQ WORDS: {}".format(high_freq_words))


def _named_entities_from_text(text: str, labels: Sequence[str]) -> Sequence[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    labels = [label.upper() for label in labels]

    wanted_ents = [ent.text for ent in doc.ents if ent.label_ in labels]
    # Cut out duplicates
    return list(set(wanted_ents))


def _plot_frequency_distribution(fdist: FreqDist) -> None:
    """TODO: Generate histogram"""
    fdist.plot()
    #matplotlib.pyplot.hist(fdist.tabulate())


def _remove_punctuation(text: str) -> str:
    return "".join([c for c in text if c not in string.punctuation])


def _remove_stopwords(tokens: Sequence[str], wordlist: Sequence[str]) -> Sequence[str]:
    return [w for w in tokens if w not in wordlist]


def _sentences_by_chapters(texts_by_chapters: Dict[str, str]) -> Dict[str, Sequence[str]]:
    """Turns a chapter_header - chapter_text representation into
    chapter_header - list_of_chapter_text_words"""
    sentences_by_chapters = dict()
    for chapter in texts_by_chapters.keys():
        sentences_by_chapters[chapter] = _tokenize_sentences(texts_by_chapters[chapter])
    return sentences_by_chapters


def _stemming(tokens: Sequence[str], stemmer) -> Sequence[str]:
    return [stemmer.stem(w) for w in tokens]


def _texts_by_chapters(html: str, titles: Sequence[str]) -> Dict[str, str]:
    """Returns a dictionary with titles as keys and str text blocks as values."""
    texts = dict()
    for title in titles:
        chapter = _chapter_from_html(html, title)
        if chapter is not None:
            texts[title] = chapter
    return texts


def _text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def _titles_from_html(html: str) -> Sequence[str]:
    """Returns the headers as a list of strings."""
    soup = BeautifulSoup(html, "html.parser")
    # Get the title(s)
    titles = [a.get_text() for a in soup.find_all("title")]
    return titles


def _tokenize_sentences(text: str) -> Sequence[str]:
    sentences = nltk.sent_tokenize("".join([char for char in text if char != "\n"]))
    return sentences


def _tokenize_text(text: str, tokenizer) -> Sequence[str]:
    return tokenizer(text)


def _word_frequency_distribution(tokens: Sequence[str]) -> FreqDist:
    return FreqDist(tokens)


if __name__ == "__main__":
    main()
