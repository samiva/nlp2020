import argparse
import logging

from typing import Any, Dict

import autosummary.autosummary as autosummary
import autosummary.config as config
import autosummary.autosummary_gui as gui
import autosummary.sumy_interface as sumy_interface

_logger = logging.getLogger(__name__)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="interface",
                                       help="Specifies whether a CLI or GUI is used.")
    subparsers.required = True

    parser_cli = subparsers.add_parser("cli", help="Command Line Interface")
    parser_cli.add_argument("source_type",
                            type=str,
                            choices=("url", "file", "dataset"),
                            help="Specifies the type of the source_path: URL or filepath.")
    parser_cli.add_argument("source_path",
                            type=str,
                            help="Filepath or URL pointing to source.")
    parser_cli.add_argument("-w",
                            dest="word_count",
                            type=int,
                            default=10,
                            help="Number of high frequency words to include for the summarization.")
    parser_cli.add_argument("-k", "--keyword",
                            dest="keyword",
                            type=str,
                            choices=["freq", "rake"],
                            default="freq",
                            help="Specifies which keyword extraction method is used.")
    parser_cli.add_argument("--ne-filter",
                            dest="ne_filter",
                            action="store_true",
                            default=False,
                            help="Filter out named entities from the keywords.")
    parser_cli.add_argument("--sumy",
                            dest="sumy",
                            action="store_true",
                            default=False,
                            help="Run summarization on sumy's summarizers as well.")
    parser_cli.add_argument("--evaluate",
                            dest="evaluate",
                            type=int,
                            default=0,
                            help="Run the summarization for the specified number of"
                                 "entries from the CNN/DailyMail dataset. Calculate"
                                 "the ROUGE2 and ROUGE3 for these summaries.")
    subparsers.add_parser("gui", help="Graphical User Interface")
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=config.LOGGING_FORMAT)
    parsed_args = _argument_parser().parse_args()
    if parsed_args.interface == "gui":
        _run_gui()
    elif parsed_args.interface == "cli":
        summary_config = {
            "source_type": parsed_args.source_type,
            "source_path": parsed_args.source_path,
            "word_count": parsed_args.word_count,
            "keyword": parsed_args.keyword,
            "ne_filter": parsed_args.ne_filter,
            "evaluate-count": parsed_args.evaluate,
            # TODO: Make configurable
            "evaluate-random": False,
            "use-sumy": parsed_args.sumy,
        }
        _run_cli(summary_config)


def _run_cli(summary_config: Dict[str, Any]):
    if summary_config["source_type"] == "dataset":
        try:
            eval_results = autosummary.evaluate_summaries(summary_config)
        except ValueError as e:
            _logger.exception(e)
            return

        for result in eval_results:
            doc_id = result[0][0]
            summary_output = result[0][1]
            eval_metrics = result[1]
            msg = "SUMMARY FOR DOC_{} (ROUGE2: p={:.3f} r={:.3f}) (ROUGE3: p={:.3f} r={:.3f}): {}"
            _logger.info(msg.format(doc_id,
                                    eval_metrics["rouge2-precision"],
                                    eval_metrics["rouge2-recall"],
                                    eval_metrics["rouge3-precision"],
                                    eval_metrics["rouge3-recall"],
                                    summary_output))
        return

    summary = autosummary.summary_by_config(summary_config)
    _logger.info("SUMMARY: {}".format(summary))
    if summary_config["use-sumy"]:
        sumy_summaries = autosummary.summary_sumy(summary_config,
                                                  sumy_interface.SUMMARIZERS.keys())
        if sumy_summaries is None:
            _logger.warning("Extraction of sumy summaries failed.")
            return
        _logger.debug(sumy_summaries)
        for summarizer, sumy_summary in sumy_summaries.items():
            _logger.info("SUMMARY [{}]: {}".format(summarizer, sumy_summary))


def _run_gui():
    gui.run()


if __name__ == "__main__":
    main()
