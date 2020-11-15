# Natural Language Processing and Text Mining course's project work

## Project 19: Automatic Summarization 2


### Foreword
This project was completed as part of studies at the University of Oulu. The code itself may not be all that useful as-is, but feel free to take a gander at what it does. As for the code quality, there was very little design at the beginning so multiple restructurings had to be applied as the project progressed. For the most part the development was stable and of reasonable quality, but towards the final commits both the commit message and code quality started to nosedive. Deadline creeped in so there was no longer enough interest in doing things right as long as it seemed to work.


### Dependencies

The program uses the following PyPI packages:
* [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [nltk](https://pypi.org/project/nltk/)
    * It may be necessary to download some parts of the nltk-data, and the surest way to make it work is to install the `book` collection.
* [rake-nltk](https://pypi.org/project/rake-nltk/)
* [spacy](https://pypi.org/project/spacy/)
    * en_core_web_sm -model (can be installed e.g. by using `python -m spacy download en_core_web_sm`)
* [sumy](https://pypi.org/project/sumy/)

For running the program using dataset (evaluation) sources, download the source materials from [here](https://github.com/morningmoni/FAR). Put the files from the output directory into the files directory within this repository. Additionally, save the ../data/all_test.txt as ../files/test_article_links.txt. Other paths and naming can be used as well, but be sure to **update the correct paths to config.py**.

### Running the program

The program execution starts from the run_autosummary.py module. It offers a CLI and a GUI interface. The CLI is used by providing the desired parameters as command line arguments whereas the GUI does not take any additional arguments.

The config.py module offers quite a few configurable parameters some of which need to be set before the program can run successfully. The `DATASET_DIRECTORY`, `DATASET_FILE` and `REFERENCE_SUMMARY_FILES` are examples of required parameters. Other parameters are mainly for tweaking the way the summarizer works.

#### Examples

Start the GUI.
`python3 run_autosummary.py gui`

CLI: Run the summarizer for a URL with high FreqDist keyword extraction method.
`python3 run_autosummary.py cli url <valid_url_here> -k freq`

CLI: Run the summarizer for a local file with RAKE keyword extraction method.
`python3 run_autosummary.py cli file path/to/local/file -k rake`

CLI: Run the summarizer for one dataset document with RAKE keyword extraction method.
`python3 run_autosummary.py cli dataset -k rake --evaluate 1`

CLI: Run the summarizer for a URL with RAKE keyword extraction method and by all of Sumy's summarizers.
`python3 run_autosummary.py cli dataset -k rake --sumy`
