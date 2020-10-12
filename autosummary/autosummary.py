
"""
This is the main module of the autosummary project. Currently it offers both
the CLI and the actual automatic summarization functionality.
"""

import argparse
import logging
import urllib.request


_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"


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
    _logger.debug(raw_html)


if __name__ == "__main__":
    main()
