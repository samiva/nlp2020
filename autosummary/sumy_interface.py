
"""Interface for making the summarization using sumy's summarization tools. The
current implementation uses the sumy's CLI as there's no proper documentation for
the python API. It was easier to implement the use via the CLI instead."""
import argparse
import logging
# TODO: Use sumy's python API instead
import subprocess

from typing import Optional

_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
_SUMMARIZERS = (
    "luhn",
    "edmundson",
    "lsa",
    "text-rank",
    "lex-rank",
    "sum-basic",
    "kl",
)


def _argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("summarizer",
                   type=str,
                   choices=_SUMMARIZERS,
                   help="Specifies which sumy's summarizer is used.")
    p.add_argument("--length", "-l",
                   type=int,
                   default=10,
                   help="Number of sentences to include in the summary.")
    p.add_argument("--url",
                   type=str,
                   required=True,
                   help="The target document's url address")
    return p


def summarize(url: str, summarizer: str, length: int = 10) -> Optional[str]:
    if summarizer not in _SUMMARIZERS:
        _logger.warning("Unsupported summarizer: '{}'".format(summarizer))
        return
    call = ["sumy", summarizer, "--length={}".format(length), "--url={}".format(url)]
    summary = subprocess.check_output(call)
    summary = summary.decode("utf-8")
    summary = summary.replace("\n", " ")
    return summary


def _main():
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOGGING_FORMAT)
    parsed_args = _argument_parser().parse_args()
    summary = summarize(parsed_args.url, parsed_args.summarizer, parsed_args.length)
    _logger.info("SUMMARY: {}".format(summary))


if __name__ == "__main__":
    _main()
