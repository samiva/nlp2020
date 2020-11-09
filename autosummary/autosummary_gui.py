
"""GUI for the automatic text summarization module."""
import enum
import logging
import os

import tkinter as tk
import tkinter.scrolledtext
import tkinter.filedialog
import tkinter.filedialog

from typing import Any, Dict, Sequence, Tuple

from . import autosummary
from . import config
from . import sumy_interface

_logger = logging.getLogger(__name__)
_WORKING_DIRECTORY = os.getcwd()


# Mapping of source type internal names to UI names
class _SourceType(enum.Enum):
    url = "URL"
    file = "Local File"
    dataset = "Dataset"


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        # Widgets that need to be accessible
        self.our_summarizer_options = None
        self.sumy_summarizer_options = None
        self.path_entry = None
        self.path_selector = None
        self.summarize_button = None
        self.result_box = None

        # Variables that need to be accessible
        self.source_type = tk.StringVar(value=_SourceType.url.name)
        self.source_path = tk.StringVar(value="")

        self.create_widgets()

    def create_left_side(self, frame: tk.Frame):
        left_side_container = tk.Frame(frame)

        summarizer_label = tk.Label(left_side_container, text="Summarizers")
        summarizer_label.pack()
        # Our summarizers
        our_summarizer_label = tk.LabelFrame(left_side_container, text="Our summarizers")
        our_summarizer_label.pack(fill="both", expand="yes")
        self.our_summarizer_options = CheckboxColumn(our_summarizer_label,
                                                     [("freq", "High FreqDist words"),
                                                      ("rake", "RAKE")])

        # Sumy's summarizers
        sumy_summarizer_label = tk.LabelFrame(left_side_container, text="Sumy's summarizers")
        sumy_summarizer_label.pack(fill="both", expand="yes")
        # The list order should remain the same so we use sorting
        options = (sorted(sumy_interface.SUMMARIZERS.items()))
        self.sumy_summarizer_options = CheckboxColumn(sumy_summarizer_label,
                                                      options)

        left_side_container.pack(side=tk.LEFT)
        self.our_summarizer_options.pack()
        self.sumy_summarizer_options.pack()

    def create_right_side(self, frame: tk.Frame):
        right_side_container = tk.Frame(frame)
        right_side_container.pack(side=tk.RIGHT)

        # Source types
        source_type_label = tk.LabelFrame(right_side_container, text="Source type")
        source_type_label.pack(fill="both", expand="yes")
        url_radiobutton = tk.Radiobutton(source_type_label,
                                         text=_SourceType.url.value,
                                         variable=self.source_type,
                                         value=_SourceType.url.name,
                                         command=self.source_type_selected)
        file_radiobutton = tk.Radiobutton(source_type_label,
                                          text=_SourceType.file.value,
                                          variable=self.source_type,
                                          value=_SourceType.file.name,
                                          command=self.source_type_selected)
        dataset_radiobutton = tk.Radiobutton(source_type_label,
                                             text=_SourceType.dataset.value,
                                             variable=self.source_type,
                                             value=_SourceType.dataset.name,
                                             command=self.source_type_selected)

        url_radiobutton.pack(side=tk.LEFT)
        file_radiobutton.pack(side=tk.LEFT)
        dataset_radiobutton.pack(side=tk.LEFT)

        # Source path selection
        path_frame = tk.Frame(right_side_container)
        path_frame.pack()
        path_label = tk.Label(path_frame, text="Path:")
        path_label.pack(side=tk.LEFT)
        self.path_entry = tk.Entry(path_frame)
        self.path_entry["textvariable"] = self.source_path
        self.path_entry.pack(side=tk.LEFT)
        self.path_selector = tk.Button(path_frame, text="Browse Files", command=self.source_type_file_browser)
        self.path_selector["state"] = "disabled"
        self.path_selector.pack(side=tk.LEFT)

        # Log box
        self.result_box = tkinter.scrolledtext.ScrolledText(right_side_container,
                                                            wrap=tk.WORD,
                                                            width=60,
                                                            height=30,
                                                            font=("Times New Roman",
                                                                  10)
                                                            )
        self.result_box.pack()
        self.result_box.configure(state='disabled')

        self.summarize_button = tk.Button(right_side_container,
                                          text="Summarize",
                                          command=self.run_summarizers)
        self.summarize_button.pack()

    def create_widgets(self):
        main_frame = tk.Frame(self)
        main_frame.pack()
        self.create_left_side(main_frame)
        self.create_right_side(main_frame)

    def run_summarizers(self):
        # Disable the button until the summarization is done
        if self.source_path.get() in (None, "", " "):
            _logger.info("No source path selected: Cannot run summarizer.")
            # TODO: Message into resultbox
            return

        if (self.source_type.get() == _SourceType.file.name and
                not self.source_path.get().endswith(".html")):
            _logger.info("Selected local file does not seem to be .html: '{}'".format(self.source_path.get()))
            # TODO: Message into resultbox
            return

        self.summarize_button["state"] = "disabled"

        # Get the selected summarizers with own/sumy division
        sumy_summarizers = self.sumy_summarizer_options.selected()
        own_summarizers = self.our_summarizer_options.selected()

        summarizer_config = {
            "source_type": self.source_type.get(),
            "source_path": self.source_path.get(),
            # TODO: Make configurable via GUI
            "word_count": config.WORD_COUNT,
            # Configured before summarization start
            "keyword": None,
            # TODO: Make these configurable via GUI
            "ne_filter": config.NE_FILTER,
            "evaluate-count": config.EVALUATE_COUNT,
            "evaluate-random": False,
        }

        for summarizer in own_summarizers:
            if summarizer_config["source_type"] == "dataset":
                summarizer_config["keyword"] = summarizer
                summarizer_config["source_path"] += "/" + autosummary.mod_config.DATASET_FILE
                try:
                    eval_results = _run_summarizer_evaluation(summarizer_config)
                except ValueError as e:
                    _logger.exception(e)
                    return

                for result in eval_results:
                    doc_id = result[0][0]
                    summary = result[0][1]
                    keywords = result[0][2]
                    named_ents = result[0][3]
                    eval_metrics = result[1]
                    msg = "SUMMARY FOR DOC_{} (ROUGE2: p={:.3f} r={:.3f}) (ROUGE3: p={:.3f} r={:.3f}): {}"
                    _logger.debug(msg.format(doc_id,
                                             eval_metrics["rouge2-precision"],
                                             eval_metrics["rouge2-recall"],
                                             eval_metrics["rouge3-precision"],
                                             eval_metrics["rouge3-recall"],
                                             summary))
            else:
                summarizer_config["keyword"] = summarizer
                summary, keywords, named_ents = _run_summarizer(summarizer_config)
            # TODO: Display these in the result_box
            _logger.info("KEYWORDS: {}".format(keywords))
            _logger.info("NAMED ENTITIES: {}".format(named_ents))
            _logger.info("SUMMARY: {}".format(summary))

        sumy_summaries = autosummary.summary_sumy(summarizer_config,
                                                  sumy_summarizers)
        if sumy_summaries is not None:
            # TODO: Code reuse
            for summarizer in sumy_summaries.keys():
                # TODO: Display these in the result_box
                if type(sumy_summaries[summarizer]) == "str":
                    _logger.info("{}: {}".format(summarizer.upper(),
                                                 sumy_summaries[summarizer]))
                else:
                    # The summary is going to be in a weird format...
                    #sumy_summaries[summarizer].append(((i, summary, ref_summaries_by_index[i]), eval_results))
                    for summary_data in sumy_summaries[summarizer]:
                        doc_id = summary_data[0]
                        summary = summary_data[1]
                        #ref_summary = summary_data[2]
                        eval_metrics = summary_data[3]
                        msg = "SUMMARY [{}] FOR DOC_{} (ROUGE2: p={:.3f} r={:.3f}) (ROUGE3: p={:.3f} r={:.3f}): {}"
                        _logger.info(msg.format(summarizer,
                                                doc_id,
                                                eval_metrics["rouge2-precision"],
                                                eval_metrics["rouge2-recall"],
                                                eval_metrics["rouge3-precision"],
                                                eval_metrics["rouge3-recall"],
                                                summary))

        # Summarization has finished. Enable the button again.
        self.summarize_button["state"] = "normal"

    def source_type_file_browser(self):
        _logger.debug("FILE BROWSER: Source type: '{}'".format(self.source_type.get()))
        if self.source_type.get() == _SourceType.file.name:
            # Local files: File browser for selecting a .html file
            source_name = tkinter.filedialog.askopenfilename(initialdir=_WORKING_DIRECTORY,
                                                             title="Select a File",
                                                             filetypes=(("HTML files", "*.html*"),
                                                                        ("All files", "*.*")))
        elif self.source_type.get() == _SourceType.dataset.name:
            # Dataset files: File browser for selecting a directory (where the .txt files lie)
            source_name = tkinter.filedialog.askdirectory(initialdir=_WORKING_DIRECTORY,
                                                          title="Select a directory",
                                                          mustexist=True)
        else:
            msg = "Source type file browser selected for unsupported source type: '{}'".format(self.source_type.get())
            raise ValueError(msg)
        # Change label contents
        self.source_path.set(source_name)

    def source_type_selected(self):
        if self.source_type.get() == _SourceType.url.name:
            self.path_selector["state"] = "disabled"
        else:
            self.path_selector["state"] = "normal"
        self.source_path.set("")


class CheckboxColumn(tk.Frame):
    # Adapted from https://www.python-course.eu/tkinter_checkboxes.php
    def __init__(self, parent: tk.Widget, option_names: Sequence[Tuple[str, str]]):
        tk.Frame.__init__(self, parent)
        self.checkboxes = dict()
        for name_tuple in option_names:
            box_val = tk.IntVar()
            # The UI name goes for the GUI
            box = tk.Checkbutton(self, text=name_tuple[1], variable=box_val)
            box.pack(side=tk.TOP, anchor=tk.W, expand=tk.YES)
            self.checkboxes[name_tuple] = box_val

    def selected(self) -> Sequence[str]:
        # Return only the internal-use name for the selected checkboxes.
        return [name_tuple[0] for name_tuple in self.checkboxes.keys() if self.checkboxes[name_tuple].get() == 1]


def run():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()


def _run_summarizer_evaluation(summarizer_config: Dict[str, Any]) -> Sequence[Tuple[Tuple[int,
                                                                                          str,
                                                                                          Sequence[str],
                                                                                          Sequence[str],
                                                                                          str],
                                                                                    Dict[str, float]]]:
    source_paths = autosummary.get_raw_source("file",
                                              summarizer_config["source_path"])
    source_paths = source_paths.split("\n")

    ref_summaries_by_index = autosummary.ref_summaries_by_indexes(source_paths,
                                                                  summarizer_config["evaluate-count"],
                                                                  summarizer_config["evaluate-random"])

    # Dataset sources are stored as urls
    summarizer_config["source_type"] = "url"
    results = []

    for i in ref_summaries_by_index.keys():
        # Go through the source_paths in order, get summaries and calculate their
        # ROUGE2 and ROUGE3 metrics based on the corresponding reference summaries.
        summarizer_config["source_path"] = source_paths[i]
        summary, keywords, processed_named_ents = _run_summarizer(summarizer_config)
        eval_results = autosummary.evaluate_summary(summary,
                                                    ref_summaries_by_index[i])
        results.append(((i, summary, keywords, processed_named_ents, ref_summaries_by_index[i]), eval_results))
    return results


def _run_summarizer(summarizer_config: Dict[str, Any]) -> Tuple[str, Sequence[str], Sequence[str]]:
    source_type = summarizer_config["source_type"]
    source_path = summarizer_config["source_path"]
    word_count = summarizer_config["word_count"]
    keyword_extraction_method = summarizer_config["keyword"]
    ne_filter = summarizer_config["ne_filter"]

    _logger.debug("Using source '{}' ({})".format(source_path,
                                                  source_type.upper()))

    title_raw, texts_by_chapters, processed_wordlists_by_chapters = autosummary.process_text(summarizer_config)

    # Preprocessed named entities
    processed_named_ents = autosummary.named_entities_from_text_chapters(texts_by_chapters,
                                                                         config.NAMED_ENTITY_TAGS)
    _logger.info("PROCESSED NAMED ENTITIES: {}".format(processed_named_ents))

    if ne_filter:
        # Filter out named entities from the keywords
        filter_words = processed_named_ents
    else:
        filter_words = None

    if keyword_extraction_method == "freq":
        keywords = autosummary.keywords_by_high_freqdist(processed_wordlists_by_chapters,
                                                         word_count,
                                                         filter_words)
    elif keyword_extraction_method == "rake":
        keywords = autosummary.keywords_by_rake(texts_by_chapters,
                                                word_count,
                                                filter_words)
    else:
        msg = "Invalid keyword extraction method: '{}'".format(keyword_extraction_method)
        raise ValueError(msg)

    sentences_by_chapters = autosummary.text_by_chapters_to_sentences_by_chapters(texts_by_chapters)
    summary = autosummary.summary_extract(summarizer_config,
                                          title_raw,
                                          sentences_by_chapters,
                                          processed_wordlists_by_chapters,
                                          processed_named_ents,
                                          keywords)
    return summary, keywords, processed_named_ents
