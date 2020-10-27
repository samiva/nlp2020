
"""GUI for the automatic text summarization module."""
import enum
import logging
import os

import tkinter as tk
import tkinter.scrolledtext
import tkinter.filedialog
import tkinter.filedialog

import config

_logger = logging.getLogger(__name__)
_WORKING_DIRECTORY = os.getcwd()


# Mapping of source type internal names to UI names
class _SourceType(enum.Enum):
    URL = "URL"
    FILE = "Local File"
    DATASET = "Dataset"


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_variables()
        self.create_widgets()

    def create_left_side(self, frame: tk.Frame):
        left_side_container = tk.Frame(frame)

        summarizer_label = tk.Label(left_side_container, text="Summarizers")
        summarizer_label.pack()
        # Our summarizers
        our_summarizer_label = tk.LabelFrame(left_side_container, text="Our summarizers")
        our_summarizer_label.pack(fill="both", expand="yes")
        high_freq_checkbox = tk.Checkbutton(our_summarizer_label, text="High frequency words")
        rake_checkbox = tk.Checkbutton(our_summarizer_label, text="RAKE")

        #Sumy's summarizers
        sumy_summarizer_label = tk.LabelFrame(left_side_container, text="Sumy's summarizers")
        sumy_summarizer_label.pack(fill="both", expand="yes")
        luhn_checkbox = tk.Checkbutton(sumy_summarizer_label, text="Luhn")
        edmundson_checkbox = tk.Checkbutton(sumy_summarizer_label, text="Edmundson")
        lsa_checkbox = tk.Checkbutton(sumy_summarizer_label, text="LSA")
        lexrank_checkbox = tk.Checkbutton(sumy_summarizer_label, text="LexRank")
        textrank_checkbox = tk.Checkbutton(sumy_summarizer_label, text="TextRank")
        sumbasic_checkbox = tk.Checkbutton(sumy_summarizer_label, text="SumBasic")
        kl_checkbox = tk.Checkbutton(sumy_summarizer_label, text="KL")
        left_side_container.pack(side=tk.LEFT)
        high_freq_checkbox.pack()
        rake_checkbox.pack()

        luhn_checkbox.pack()
        edmundson_checkbox.pack()
        lsa_checkbox.pack()
        lexrank_checkbox.pack()
        textrank_checkbox.pack()
        sumbasic_checkbox.pack()
        kl_checkbox.pack()

    def create_right_side(self, frame: tk.Frame):
        right_side_container = tk.Frame(frame)
        right_side_container.pack(side=tk.RIGHT)

        # Source types
        source_type_label = tk.LabelFrame(right_side_container, text="Source type")
        source_type_label.pack(fill="both", expand="yes")
        url_radiobutton = tk.Radiobutton(source_type_label,
                                         text=_SourceType.URL.value,
                                         variable=self.source_type,
                                         value=_SourceType.URL.name,
                                         command=self.source_type_selected)
        file_radiobutton = tk.Radiobutton(source_type_label,
                                          text=_SourceType.FILE.value,
                                          variable=self.source_type,
                                          value=_SourceType.FILE.name,
                                          command=self.source_type_selected)
        dataset_radiobutton = tk.Radiobutton(source_type_label,
                                             text=_SourceType.DATASET.value,
                                             variable=self.source_type,
                                             value=_SourceType.DATASET.name,
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
        result_box = tkinter.scrolledtext.ScrolledText(right_side_container,
                                                       wrap=tk.WORD,
                                                       width=40,
                                                       height=10)
        result_box.pack()
        result_box.configure(state='disabled')

        summarize_button = tk.Button(right_side_container, text="Summarize")
        summarize_button.pack()

    def create_variables(self):
        self.source_type = tk.StringVar(value=_SourceType.URL.name)
        self.source_path = tk.StringVar(value="")

    def create_widgets(self):
        main_frame = tk.Frame(self)
        main_frame.pack()
        self.create_left_side(main_frame)
        self.create_right_side(main_frame)

    def source_type_file_browser(self):
        _logger.debug("FILE BROWSER: Source type: '{}'".format(self.source_type.get()))
        if self.source_type.get() == _SourceType.FILE.name:
            # Local files: File browser for selecting a .html file
            source_name = tkinter.filedialog.askopenfilename(initialdir=_WORKING_DIRECTORY,
                                                             title="Select a File",
                                                             filetypes=(("HTML files", "*.html*"),
                                                                        ("All files", "*.*")))
        elif self.source_type.get() == _SourceType.DATASET.name:
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
        if self.source_type.get() == _SourceType.URL.name:
            self.path_selector["state"] = "disabled"
        else:
            self.path_selector["state"] = "normal"
        self.source_path.set("")


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=config.LOGGING_FORMAT)
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
