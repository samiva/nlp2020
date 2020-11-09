
"""
This is the main module of the autosummary project. It offers the API for the
summarizers and their evaluation. The summarization logic for the high freqdist
word summarizer is also included in this library.
"""

import logging
import random
import re
import urllib.request

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot
import nltk.tokenize
import spacy

from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from rake_nltk import Rake

from . import sumy_interface
from . import config as mod_config


_logger = logging.getLogger(__name__)


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


def _element_count_map(elem_list: List[Any]) -> Dict[Any, int]:
    """Create a dictionary of VALUE - COUNT pairs for every unique element within
    a list of elements. The count indicates the number of times the specific
    element appears in the list."""
    count_map = dict()
    for elem in elem_list:
        if elem in count_map:
            # If the element has been already added to the dict, increase the count.
            count_map[elem] += 1
        else:
            # The element wasn't yet in the dict, add it there.
            count_map[elem] = 1
    return count_map


def _element_overlap(a: List[Any], b: List[Any]) -> int:
    """Calculate the number of common elements between two lists of elements."""
    overlap = 0
    # Create VALUE - COUNT maps for the lists
    a_map = _element_count_map(a)
    b_map = _element_count_map(b)
    for elem in a_map:
        if elem in b_map:
            # If an element that appears in a also appears in b, add the lower
            # count (meaning the count of common appearances) to the number of
            # overlaps.
            overlap += min(a_map[elem], b_map[elem])
    return overlap


def ref_summaries_by_indexes(source_paths: Sequence[str],
                             evaluate_count: int,
                             random_indexes: bool = False) -> Dict[int, str]:
    """Extract the reference summaries for a number of documents equal to evaluate_count.
    There is an assumption that the document indexes are numbered from 0..N. Pick
    the first evaluate_count document indexes if random_indexes is False. Otherwise
    pick them at random from between 0 and the number of elements in source_paths."""
    if evaluate_count == 0:
        msg = "Invalid evaluate count for ref summary extraction: '{}'.".format(evaluate_count)
        raise ValueError(msg)

    if evaluate_count > len(source_paths):
        msg = "Evaluate count ({}) exceeded the number of source elements ({}).".format(evaluate_count,
                                                                                        len(source_paths))
        raise ValueError(msg)

    if random_indexes:
        # Pick random document indexes for the summarization
        indexes = []
        for i in range(evaluate_count):
            # Select a random index from the available document indexes
            indexes.append(random.randint(0, len(source_paths) + 1))
    else:
        # Pick the indexes from the beginning
        indexes = [i for i in range(evaluate_count)]
    return ref_summaries_for_indexes(indexes)


def evaluate_summaries(config: Dict[str, Any]) -> Sequence[Tuple[Tuple[int, str, str], Dict[str, float]]]:
    """Read the reference summaries for the dataset. Pick document indexes for
    evaluation (either by random, if evaluate-random is set to True, or in order
    from the beginning otherwise). Run the summarization algorithm on the
    documents specified by the indexes. Evaluate the summaries by calculating
    ROUGE2 and ROUGE3 metrics for the summaries based on the corresponding
    reference summaries."""

    # Get a list of the dataset source urls
    source_paths = get_raw_source("file", config["source_path"])
    source_paths = source_paths.split("\n")

    ref_summaries_by_index = ref_summaries_by_indexes(source_paths,
                                                      config["evaluate-count"],
                                                      config["evaluate-random"])

    summary_config = config
    # Dataset sources are stored as urls
    summary_config["source_type"] = "url"
    results = []

    for i in ref_summaries_by_index.keys():
        # Go through the source_paths in order, get summaries and calculate their
        # ROUGE2 and ROUGE3 metrics based on the corresponding reference summaries.
        summary_config["source_path"] = source_paths[i]
        summary = summary_by_config(summary_config.copy())
        eval_results = evaluate_summary(summary, ref_summaries_by_index[i])
        results.append(((i, summary, ref_summaries_by_index[i]), eval_results))
    return results


def evaluate_summary(summary: str, ref_summary: str) -> Dict[str, float]:
    """Returns the ROUGE2 and ROUGE3 (precision & recall) metrics for the summary."""
    # ROUGE 2
    summary_bigrams = [a for a in nltk.bigrams(summary)]
    ref_summary_bigrams = [b for b in nltk.bigrams(ref_summary)]
    bigram_overlap = _element_overlap(summary_bigrams, ref_summary_bigrams)
    rouge2_precision = bigram_overlap / len(summary_bigrams)
    rouge2_recall = bigram_overlap / len(ref_summary_bigrams)

    # ROUGE 3
    summary_trigrams = [a for a in nltk.trigrams(summary)]
    ref_summary_trigrams = [b for b in nltk.trigrams(ref_summary)]
    trigram_overlap = _element_overlap(summary_trigrams, ref_summary_trigrams)
    rouge3_precision = trigram_overlap / len(summary_trigrams)
    rouge3_recall = trigram_overlap / len(ref_summary_trigrams)
    eval_results = {
        "rouge2-precision": rouge2_precision,
        "rouge2-recall": rouge2_recall,
        "rouge3-precision": rouge3_precision,
        "rouge3-recall": rouge3_recall,
    }
    return eval_results


def _freq_dist_for_word_lists(processed_wordlists_by_chapters: Sequence[Tuple[str, Sequence[Sequence[str]]]],
                              filter_words: Optional[Sequence[str]] = None) -> FreqDist:
    """Turn the list of (header, wordlist) representation into a list of words.
    Run the word frequency distribution for these words. If a filter wordlist is
    given, filter out those words from the included words."""
    words = []
    for header, sentences in processed_wordlists_by_chapters:
        for sentence in sentences:
            # Gather up a list of words
            if filter_words is not None:
                # Filter is specified. Filter out the specified words.
                sentence = list(filter(lambda x: x not in filter_words, sentence))
            words.extend(sentence)
    return _word_frequency_distribution(words)


def get_raw_source(type_: str, path: str) -> str:
    """Handle the source fetching: get the raw HTML out of filepath or url"""
    def _read_file(filepath: str) -> str:
        with open(filepath, "r") as f:
            contents = f.read()
        return contents

    def _read_url(url: str) -> str:
        # TODO: Proxy?
        hdr = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
        request = urllib.request.Request(url=url, headers=hdr)
        return urllib.request.urlopen(request).read()

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
    # TODO: More thorough stripping (e.g. r"\n\t ")
    headers = [a.get_text().strip("¶") for a in soup.find_all(re.compile('^h[1-6]$'))]
    # Remove unwanted chapters
    headers = [a for a in headers if a.upper() not in mod_config.UNWANTED_CHAPTERS]
    return headers


def _highest_freq_words(freq_dist: FreqDist, word_count: int) -> Sequence[str]:
    return [a[0] for a in freq_dist.most_common(word_count)]


def keywords_by_high_freqdist(processed_wordlists_by_chapters: Sequence[Tuple[str, Sequence[Sequence[str]]]],
                              word_count: int,
                              filter_words: Sequence[str] = None) -> Sequence[str]:
    """Calculate a frequency distribution for the words, plot it, and return the
     word_count words with the highest word frequency."""
    # Use words with highest frequency distribution for keyword selection
    freq_dist = _freq_dist_for_word_lists(processed_wordlists_by_chapters,
                                          filter_words)
    _plot_frequency_distribution(freq_dist)
    return _highest_freq_words(freq_dist, word_count)


def keywords_by_rake(texts_by_chapters: Sequence[Tuple[str, str]],
                     word_count: int,
                     filter_words: Optional[Sequence[str]] = None) -> Sequence[str]:
    """Extract keywords from the raw complete text (by appending the chapter-divided
    text blocks into a complete text block) using RAKE. The RAKE-ranked keywords shall
    be preprocessed and duplicates shall be removed. If filter_words is provided,
    those words will be filtered out from the list of keywords. Only the word_count
    most highly ranked keywords shall be returned."""
    complete_text_by_chapters = []
    for header, text_block in texts_by_chapters:
        complete_text_by_chapters.append(text_block)

    r = Rake(stopwords=mod_config.STOP_WORDS,
             punctuations=mod_config.PUNCTUATION,
             max_length=1,
             min_length=1)
    # Extract keywords from the text_block
    r.extract_keywords_from_sentences(complete_text_by_chapters)

    # Get list of ranked keywords (highest-lowest)
    keywords = r.ranked_phrases
    _logger.debug("Raw RAKE keywords: {}...".format(keywords[:20]))

    # Preprocess the keywords
    keywords = _preprocess_words(keywords)
    keywords = _remove_duplicates(keywords)

    if filter_words is not None:
        # Filter is specified. Filter out the specified words from the keywords.
        keywords = list(filter(lambda x: x not in filter_words, keywords))
    # Return (possibly filtered) list of preprocessed keywords in ranked order
    return keywords[:word_count]


def _lemmatize_tokens(tokens: Sequence[str]) -> Sequence[str]:
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in tokens]


def named_entities_from_text_chapters(texts_by_chapters: Sequence[Tuple[str, str]],
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


def _parse_ref_summaries(raw_summaries: Sequence[str]) -> Dict[int, Sequence[str]]:
    """This is more of a scraper currently. There is a strong expectation that
    indexes and facets are ordered properly."""
    doc_indexes = []
    ref_summaries = dict()

    for line in raw_summaries:
        if line.startswith("idx: "):
            # Specifies document index
            doc_indexes.append(int(line[5:]))
        if line.startswith("Facet-"):
            # Assume that the line contains reference summary for the latest index.
            # The summary begins after the ': ', so pick the summary after that.
            ref_summary = line[line.find(":") + 2:]
            if doc_indexes[-1] not in ref_summaries:
                # First facet for the document index
                ref_summaries[doc_indexes[-1]] = [ref_summary]
            else:
                # Facet for the document index already exists, append to it.
                ref_summaries[doc_indexes[-1]].append(ref_summary)
    return ref_summaries


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
    tokens_without_stopwords = _remove_stopwords(tokens, mod_config.STOP_WORDS)
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


def process_text(config: Dict[str, Any]) -> Tuple[str,
                                                  Sequence[Tuple[str, str]],
                                                  Sequence[Tuple[str, Sequence[Sequence[str]]]]]:
    """Extract the raw title, raw texts by chapters (headers) and also the
    preprocessed wordlists (lists of lists [sentences] of words). Works for both
    locally stored files and urls."""
    source_type = config["source_type"]
    source_path = config["source_path"]

    _logger.debug("Using source '{}' ({})".format(source_path, source_type.upper()))

    raw_html = get_raw_source(source_type, source_path)

    title_raw = _title_from_html(raw_html)
    headers_raw = _headers_from_html(raw_html)
    _logger.debug("HEADERS: {}".format(headers_raw))

    texts_by_chapters = _texts_by_chapters(raw_html, headers_raw)
    sentences_by_chapters = text_by_chapters_to_sentences_by_chapters(texts_by_chapters)
    _logger.debug("SENTENCES BY CHAPTERS: {}".format(sentences_by_chapters))

    # Preprocessing
    processed_wordlists_by_chapters = _preprocess_sentences(sentences_by_chapters)
    _logger.debug("PROCESSED LISTS OF WORDS BY CHAPTERS: {}".format(processed_wordlists_by_chapters))

    return title_raw, texts_by_chapters, processed_wordlists_by_chapters


def _read_reference_summaries() -> Dict[int, str]:
    """Reads the reference summaries for the documents and returns a dictionary
    with document indexes as the keys and reference summaries as values."""
    ref_summaries = dict()
    for ref_summary_file in mod_config.REFERENCE_SUMMARY_FILES:
        complete_file_path = mod_config.DATASET_DIRECTORY + ref_summary_file
        with open(complete_file_path, "r") as f:
            raw_summaries = f.readlines()
            parsed_summaries = _parse_ref_summaries(raw_summaries)
            ref_summaries.update(parsed_summaries)
    return ref_summaries


def ref_summaries_for_indexes(wanted_indexes: Sequence[int]) -> Dict[int, str]:
    """Returns a dictionary where each document index has a reference summary as
    a single string."""
    ref_summaries_all = _read_reference_summaries()
    ref_summaries = dict()
    ref_summary_indexes = ref_summaries_all.keys()

    for i in wanted_indexes:
        if i not in ref_summary_indexes:
            _logger.warning("Wanted index '{}' not in reference summary indexes!".format(i))
            continue
        # TODO: Could do something more clever with this
        # Take only Facet-0!
        ref_summaries[i] = ref_summaries_all[i][0]
    return ref_summaries


def _remove_duplicates(words: Sequence[str]) -> Sequence[str]:
    result = []
    for w in words:
        if w not in result:
            result.append(w)
    return result


def _remove_punctuation(text: str) -> str:
    return "".join([c for c in text if c not in mod_config.PUNCTUATION])


def _remove_stopwords(tokens: Sequence[str], wordlist: Sequence[str]) -> Sequence[str]:
    return [w for w in tokens if w not in wordlist]


def text_by_chapters_to_sentences_by_chapters(texts_by_chapters: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, Sequence[str]]]:
    """Turns a (chapter_header, chapter_text) representation into
    (chapter_header, list_of_chapter_text_words)."""
    sentences_by_chapters = []
    for header, chapter in texts_by_chapters:
        sentences_by_chapters.append((header, _tokenize_sentences(chapter)))
    return sentences_by_chapters


def _stemming(tokens: Sequence[str], stemmer) -> Sequence[str]:
    return [stemmer.stem(w) for w in tokens]


def _summary_build(raw_sentences: Sequence[Tuple[str, Sequence[str]]],
                   title: Tuple[str, Sequence[str]],
                   processed_sentences: Sequence[Tuple[str, Sequence[Sequence[str]]]],
                   keywords: Sequence[str],
                   named_ents: Sequence[str]) -> str:
    """
    :param raw_sentences: Unprocessed sentences grouped by header
    :param title: raw title and processed title (as sequence of words)
    :param processed_sentences: Processed sentences split into words and grouped by header
    :param keywords: List of processed high frequency words
    :param named_ents: List of processed named entities
    :return:
    """
    # Sentences as ((index, index2), strings) tuple
    summary_sentences = []
    for word in keywords:
        result = _summary_sentence_for_word(raw_sentences,
                                            title,
                                            processed_sentences,
                                            word,
                                            named_ents,
                                            summary_sentences)
        if result is None:
            continue
        summary_index, summary_sentence = result
        index_chapter, index_sentence = summary_index
        summary_sentences.append(((index_chapter, index_sentence), summary_sentence))
    summary_sentences.sort()
    # TODO: Fix
    """if ((-1, -1), title[0]) in summary_sentences:
        # Title is (-1, -1) so it's always the first one
        summary_sentences[0][1][0] = summary_sentences[0][1][0].upper() + "\n\n"""
    return " ".join([a[1] for a in summary_sentences])


def summary_by_config(config: Dict[str, Any]) -> str:
    """Run the entirety of the summarization process straight from the step of
    reading out the source, turning it into workable data structures, preprocessing
    it, tagging the named entities, extracting the keywords right to running the
    summarization logic."""
    source_type = config["source_type"]
    source_path = config["source_path"]
    word_count = config["word_count"]
    keyword_extraction_method = config["keyword"]
    ne_filter = config["ne_filter"]

    _logger.debug("Using source '{}' ({})".format(source_path,
                                                  source_type.upper()))

    title_raw, texts_by_chapters, processed_wordlists_by_chapters = process_text(config)

    # Preprocessed named entities
    processed_named_ents = named_entities_from_text_chapters(texts_by_chapters,
                                                             mod_config.NAMED_ENTITY_TAGS)
    _logger.info("PROCESSED NAMED ENTITIES: {}".format(processed_named_ents))

    if ne_filter:
        # Filter out named entities from the keywords
        filter_words = processed_named_ents
    else:
        filter_words = None

    if keyword_extraction_method == "freq":
        keywords = keywords_by_high_freqdist(processed_wordlists_by_chapters,
                                             word_count,
                                             filter_words)
    elif keyword_extraction_method == "rake":
        keywords = keywords_by_rake(texts_by_chapters,
                                    word_count,
                                    filter_words)
    else:
        msg = "Invalid keyword extraction method: '{}'".format(keyword_extraction_method)
        raise ValueError(msg)

    sentences_by_chapters = text_by_chapters_to_sentences_by_chapters(texts_by_chapters)
    return summary_extract(config,
                           title_raw,
                           sentences_by_chapters,
                           processed_wordlists_by_chapters,
                           processed_named_ents,
                           keywords)


def summary_extract(config: Dict[str, Any],
                    title_raw: str,
                    sentences_by_chapters: Sequence[Tuple[str, Sequence[str]]],
                    processed_wordlists_by_chapters: Sequence[Tuple[str, Sequence[Sequence[str]]]],
                    processed_named_ents: Sequence[str],
                    keywords: Sequence[str]) -> str:

    ne_filter = config["ne_filter"]

    # Title representation (raw, preprocessed)
    title = (title_raw, _tokenize_text(title_raw.lower(), nltk.tokenize.word_tokenize))

    _logger.info("KEYWORDS: {}".format(keywords))
    if ne_filter:
        # Named Entity filtering is used for the keywords
        for word in keywords:
            # Debug check: Did the named entity filtering work?
            if word in processed_named_ents:
                raise ValueError("Found named entity from high freq words: '{}'.".format(word))
    return _summary_build(sentences_by_chapters,
                          title,
                          processed_wordlists_by_chapters,
                          keywords,
                          processed_named_ents)


def _summary_sentence_for_word(raw_sentences: Sequence[Tuple[str, Sequence[str]]],
                               title: Tuple[str, Sequence[str]],
                               processed_sentences: Sequence[Tuple[str, Sequence[Sequence[str]]]],
                               word: str,
                               named_ents: Sequence[str],
                               already_selected: List[Tuple[Tuple[int, int], str]]) \
        -> Optional[Tuple[Tuple[int, int], str]]:
    """Goes through the preprocessed sentences (chapter by chapter) and apply
    the summarization logic by selecting a sentence for the summary (or not).
    Return the sentence and it's 2D index (chapter_i, sentence_i) if...
    * The sentence has not been selected already (check already_selected indexes)
    * The sentence contains the given keyword and...
        - is located in the abstract chapter
        - the sentence also contains a named entity
    * If no sentence fulfills the previous criteria, select the first (not yet
    selected) sentence that contains the keyword.
    * Return none if even this final criteria cannot be fulfilled."""
    summary_sentence_candidate = None
    already_selected_indexes = [a[0] for a in already_selected]
    if (-1, -1) not in already_selected_indexes:
        # Title is not yet added. The index (-1, -1) is considered the title.
        if word in title[1]:
            return (-1, -1), title[0]

    for i, (chapter_header, sentences) in enumerate(processed_sentences):
        # i for indexing sentences: Order of summary sentences!
        for j, sentence in enumerate(sentences):
            if word not in sentence:
                # No keyword in the sentence, skip it!
                continue
            if (i, j) in already_selected_indexes:
                # The sentence is selected already, skip it!
                continue
            if chapter_header.upper() == "ABSTRACT":
                # Keyword in a sentence within abstract, add it!
                _logger.info("SUMMARY - KEYWORD IN ABSTRACT SENTENCE")
                return (i, j), raw_sentences[i][1][j]
            if _named_entity_in_sentence(sentence, named_ents):
                # Keyword and named entity in the sentence, add it!
                _logger.info("SUMMARY - KEYWORD AND NAMED ENTITY IN A SENTENCE")
                return (i, j), raw_sentences[i][1][j]
            if summary_sentence_candidate is None:
                # No "First sentence not in abstract that doesn't contain a
                # named entity" - candidate currently. This is the one.
                summary_sentence_candidate = (i, j), raw_sentences[i][1][j]

    if summary_sentence_candidate is None:
        _logger.info("SUMMARY - NO CANDIDATE FOUND FOR '{}'".format(word))
    else:
        _logger.info("SUMMARY - FIRST SENTENCE WITH KEYWORD")

    # The keyword was not in title/abstract and no sentence contained both the
    # keyword and a named entity. Returning the first not yet included sentence
    # that contained the keyword (if it exists).
    return summary_sentence_candidate


def summary_sumy(config: Dict[str, Any],
                 summarizers: Sequence[str],
                 summary_length: int = 10) -> Optional[Dict[str,
                                                            Union[str,
                                                                  Sequence[Tuple[int, str, str, Dict[str,
                                                                                                     float]]]]]]:
    """Run specified sumy summarizers for the specified document. Return a dictionary
    of summarizer_name: output_summary key: value pairs."""
    # TODO: Consider a less silly data structure to return

    sumy_summaries = dict()

    if config["source_type"] == "url":
        # Run summarization on sumy's summarizers
        for summarizer in summarizers:
            sumy_summaries[summarizer] = sumy_interface.summarize_by_url(config["source_path"],
                                                                         summarizer,
                                                                         summary_length)
    elif config["source_type"] == "dataset":
        # TODO: Reuse the summary evaluation code - the following piece is mostly duplicate
        # Get a list of the dataset source urls
        if not config["source_path"].endswith(mod_config.DATASET_FILE):
            config["source_path"] += "/" + mod_config.DATASET_FILE
        source_paths = get_raw_source("file", config["source_path"])
        source_paths = source_paths.split("\n")

        ref_summaries_by_index = ref_summaries_by_indexes(source_paths,
                                                          config["evaluate-count"],
                                                          config["evaluate-random"])

        for summarizer in summarizers:
            for i in ref_summaries_by_index.keys():
                # Go through the source_paths in order, get summaries and calculate their
                # ROUGE2 and ROUGE3 metrics based on the corresponding reference summaries.
                summary = sumy_interface.summarize_by_url(source_paths[i],
                                                          summarizer,
                                                          summary_length)
                if summary in (None, "", " "):
                    # TODO: Better handling?
                    # Sumy's summarizer was not able to extract a summary: Skip it.
                    sumy_summaries[summarizer] = None
                    continue
                eval_results = evaluate_summary(summary, ref_summaries_by_index[i])
                if summarizer not in sumy_summaries.keys():
                    # First summary for the summarizer
                    sumy_summaries[summarizer] = [((i, summary, ref_summaries_by_index[i]), eval_results)]
                else:
                    # At least one summary already added: we can append to the list.
                    sumy_summaries[summarizer].append((i, summary, ref_summaries_by_index[i], eval_results))
    else:
        # TODO: Support for local files
        msg = "Source type '{}' not supported for sumy summaries.".format(config["source_type"])
        _logger.warning(msg)
        return None
    return sumy_summaries


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
