
"""
This is the main module of the autosummary project. Currently it offers both
the CLI and the actual automatic summarization functionality.
"""

import argparse
import logging
import re
import string
import urllib.request

from typing import Dict, Sequence, Tuple, Optional

import matplotlib.pyplot
import nltk.tokenize
import spacy

from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.corpus import stopwords


_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
_NAMED_ENTITY_TAGS = ("PERSON", "ORG")
# string.punctuation doesn't consider these different quotation marks by default
_PUNCTUATION = string.punctuation + r'“' + r'”'
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


def _freq_dist_for_word_lists(processed_wordlists_by_chapters: Sequence[Tuple[str, Sequence[Sequence[str]]]]) -> FreqDist:
    """Turn the list of (header, wordlist) representation into a list of words.
    Run the word frequency distribution for these words."""
    words = []
    for header, sentences in processed_wordlists_by_chapters:
        for sentence in sentences:
            # Gather up a list of words
            words.extend(sentence)
    return _word_frequency_distribution(words)


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
    # Get all headers in the order of appearance
    # https://stackoverflow.com/questions/45062534/how-to-grab-all-headers-from-a-website-using-beautifulsoup
    headers = [a.get_text().strip("¶") for a in soup.find_all(re.compile('^h[1-6]$'))]
    return headers


def _highest_freq_words(freq_dist: FreqDist, word_count: int) -> Sequence[str]:
    return [a[0] for a in freq_dist.most_common(word_count)]


def _lemmatize_tokens(tokens: Sequence[str]) -> Sequence[str]:
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in tokens]


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
    title_raw = _title_from_html(raw_html)
    headers_raw = _headers_from_html(raw_html)
    _logger.debug("HEADERS: {}".format(headers_raw))
    texts_by_chapters = _texts_by_chapters(raw_html, headers_raw)
    sentences_by_chapters = _sentences_by_chapters(texts_by_chapters)
    _logger.debug("SENTENCES BY CHAPTERS: {}".format(sentences_by_chapters))

    if sentences_by_chapters is None:
        _logger.info("No text found.")
        return

    # Preprocessing
    processed_wordlists_by_chapters = _preprocess_sentences(sentences_by_chapters)
    _logger.debug("PROCESSED LISTS OF WORDS BY CHAPTERS: {}".format(processed_wordlists_by_chapters))

    # Frequency distribution
    freq_dist = _freq_dist_for_word_lists(processed_wordlists_by_chapters)
    _plot_frequency_distribution(freq_dist)

    title = (title_raw, _tokenize_text(title_raw.lower(), nltk.tokenize.word_tokenize))
    processed_named_ents = _named_entities_from_text_chapters(texts_by_chapters,
                                                              _NAMED_ENTITY_TAGS)
    _logger.info("PROCESSED NAMED ENTITIES: {}".format(processed_named_ents))
    high_freq_words = _highest_freq_words(freq_dist, word_count)
    _logger.info("HIGH FREQ WORDS: {}".format(high_freq_words))


def _named_entities_from_text_chapters(texts_by_chapters: Sequence[Tuple[str, str]],
                                       labels: Sequence[str]) -> Sequence[str]:
    labels = [label.upper() for label in labels]
    nlp = spacy.load("en_core_web_sm")

    wanted_ents = []
    for header, chapter in texts_by_chapters:
        doc = nlp(chapter)
        wanted_ents.extend([ent.text for ent in doc.ents if ent.label_ in labels])
    # Preprocess
    processed_wanted_ents = _preprocess_words(wanted_ents)
    # Cut out duplicates
    return list(set(processed_wanted_ents))


def _named_entity_in_sentence(sentence: Sequence[str],
                              named_ents: Sequence[str]) -> bool:
    for word in sentence:
        if word in named_ents:
            return True
    return False


def _plot_frequency_distribution(fdist: FreqDist) -> None:
    """TODO: Generate histogram"""
    fdist.plot()
    #matplotlib.pyplot.hist(fdist.tabulate())


def _preprocess_sentences(sentences_by_chapters: Sequence[Tuple[str, Sequence[str]]]) \
        -> Sequence[Tuple[str, Sequence[Sequence[str]]]]:
    """Go through chapters header at a time. For each header, go through the
    chapter sentences one sentence at a time. Preprocess each sentence into
    a list of preprocessed words."""
    processed_words_by_chapters = []
    for header, sentences in sentences_by_chapters:
        # Process every chapter by going through its sentences
        processed_words_by_chapter = []
        for sentence in sentences:
            # Process every sentence within the chapter into a list of words
            processed_sentence = _preprocess_text(sentence)
            processed_words_by_chapter.append(processed_sentence)
        processed_words_by_chapters.append((header, processed_words_by_chapter))
    return processed_words_by_chapters


def _preprocess_text(raw_text: str) -> Sequence[str]:
    text = raw_text.lower()
    text = _remove_punctuation(text)
    tokens = _tokenize_text(text, nltk.tokenize.word_tokenize)
    tokens_without_stopwords = _remove_stopwords(tokens, _STOP_WORDS)
    # stemmed_tokens = _stemming(tokens_without_stopwords, nltk.PorterStemmer())
    lemmatized_tokens = _lemmatize_tokens(tokens_without_stopwords)
    return lemmatized_tokens


def _preprocess_words(words: Sequence[str]) -> Sequence[str]:
    processed_words = []
    for word in words:
        processed_word = word.lower()
        processed_word = _remove_punctuation(processed_word)
        # TODO: Sometimes named entities happen to be stopwords for some reason
        # processed_word = _remove_stopwords([processed_word], _STOP_WORDS)[0]
        processed_word = _lemmatize_tokens([processed_word])[0]
        processed_words.append(processed_word)
    return processed_words


def _remove_punctuation(text: str) -> str:
    return "".join([c for c in text if c not in _PUNCTUATION])


def _remove_stopwords(tokens: Sequence[str], wordlist: Sequence[str]) -> Sequence[str]:
    return [w for w in tokens if w not in wordlist]


def _sentences_by_chapters(texts_by_chapters: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, Sequence[str]]]:
    """Turns a (chapter_header, chapter_text) representation into
    (chapter_header, list_of_chapter_text_words)."""
    sentences_by_chapters = []
    for header, chapter in texts_by_chapters:
        sentences_by_chapters.append((header, _tokenize_sentences(chapter)))
    return sentences_by_chapters


def _stemming(tokens: Sequence[str], stemmer) -> Sequence[str]:
    return [stemmer.stem(w) for w in tokens]


def _summarize(raw_sentences: Sequence[str],
               title: Tuple[str, Sequence[str]],
               stemmed_sentences: Sequence[Tuple[str, Sequence[str]]],
               stemmed_high_freq_words: Sequence[str],
               named_ents: Sequence[str]) -> Sequence[str]:
    """
    :param raw_sentences: Unstemmed sentences
    :param title: raw title and stemmed title (as sequence of str words)
    :param stemmed_sentences:
    :param stemmed_high_freq_words:
    :param named_ents: Stemmed named entities
    :return:
    """
    # Sentences as strings
    summary_sentences = []
    # Indexes of the sentences
    summary_indexes = []
    for word in stemmed_high_freq_words:
        summary_index, summary_sentence = _summarize_for_word(raw_sentences,
                                                              title,
                                                              stemmed_sentences,
                                                              word,
                                                              named_ents,
                                                              summary_indexes)
        if summary_sentence is not None:
            summary_sentences.append(summary_sentence)
            summary_indexes.append(summary_index)
    return summary_sentences


def _summarize_for_word(raw_sentences: Sequence[str],
                        title: Tuple[str, Sequence[str]],
                        stemmed_sentences: Sequence[Tuple[str, Sequence[str]]],
                        word: str,
                        named_ents: Sequence[str],
                        already_selected: Sequence[int]) -> Optional[Tuple[int, str]]:
    summary_sentence_candidate = None
    if -1 not in already_selected:
        # Title is not yet added (-1 is considered the title)
        if word in title[1]:
            return -1, title[0]

    for i, sentences_repr in enumerate(stemmed_sentences):
        sentence_chapter = sentences_repr[0]
        sentence = sentences_repr[1]
        # i for indexing sentences: Order of summary sentences!
        if word in sentence:
            if sentence_chapter.upper() == "ABSTRACT":
                # High freq word in a sentence within abstract
                if i not in already_selected:
                    # The sentence is not yet in the summary, add it!
                    _logger.info("SUMMARY - SENTENCE IN ABSTRACT")
                    return i, raw_sentences[i]
            if _named_entity_in_sentence(sentence, named_ents):
                # High freq word and named entity in the sentence
                if i not in already_selected:
                    # The sentence is not yet in the summary, add it!
                    _logger.info("SUMMARY - SENTENCE CONTAINS NAMED ENTITY")
                    return i, raw_sentences[i]
            if i not in already_selected and summary_sentence_candidate is None:
                # No "First sentence not in title/abstract that doesn't contain
                # named entity" - candidate currently. This is the one.
                summary_sentence_candidate = i, raw_sentences[i]

    # There was no high freq word in title/abstract and no sentence contained
    # both a high freq word and a named entity. Returning the first not yet
    # included sentence that contained a high freq word.
    return summary_sentence_candidate


def _texts_by_chapters(html: str, headers: Sequence[str]) -> Sequence[Tuple[str, str]]:
    """Returns a sequence of (header, text block)."""
    texts = []
    for header in headers:
        chapter = _chapter_from_html(html, header)
        if chapter is not None:
            texts.append((header, chapter))
    return texts


def _title_from_html(html: str) -> str:
    """Returns the title of the page."""
    soup = BeautifulSoup(html, "html.parser")
    # Get the title(s)
    title = soup.find("title").get_text()
    return title


def _tokenize_sentences(text: str) -> Sequence[str]:
    sentences = nltk.sent_tokenize("".join([char for char in text if char != "\n"]))
    return sentences


def _tokenize_text(text: str, tokenizer) -> Sequence[str]:
    return tokenizer(text)


def _word_frequency_distribution(tokens: Sequence[str]) -> FreqDist:
    return FreqDist(tokens)


if __name__ == "__main__":
    main()
