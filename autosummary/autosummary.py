
"""
This is the main module of the autosummary project. Currently it offers both
the CLI and the actual automatic summarization functionality.
"""

import argparse
import logging
import re
import urllib.request

from typing import Sequence

from bs4 import BeautifulSoup

_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"


def _abstract_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for header in soup.find_all(re.compile('^h[1-6]$')):
        if header.get_text().upper() == "ABSTRACT":
            # Get the text of the following item after the abstract header
            return header.next_sibling.get_text()


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
    abstract = _abstract_from_html(raw_html)
    _logger.debug("ABSTRACT: {}".format(abstract))


def _titles_from_html(html: str) -> Sequence[str]:
    """Returns the title(s) and headers as a list."""
    soup = BeautifulSoup(html, "html.parser")
    # Get the title(s)
    titles = [a.get_text() for a in soup.find_all("title")]
    # Get all headers in the order of appearance
    # https://stackoverflow.com/questions/45062534/how-to-grab-all-headers-from-a-website-using-beautifulsoup
    titles.extend(a.get_text().strip("¶") for a in soup.find_all(re.compile('^h[1-6]$')))
    return titles


if __name__ == "__main__":
    main()
