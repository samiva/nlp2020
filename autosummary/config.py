import string

from nltk.corpus import stopwords

DATASET_DIRECTORY = "../nlp2020/files/"
# Note: Remember to set the directory before using autosummary!
DATASET_FILE = "test_article_links.txt"
EVALUATE_COUNT = 1
LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
# Spacy tags for the named entities we're interested in using for the summarization logic
NAMED_ENTITY_TAGS = ("PERSON", "ORG")
# Specifies whether named entities will be filtered out of keywords.
NE_FILTER = False
# Enable this if you are using CLI and want to see this. Does not work with GUI.
PLOT_FREQ_DIST = False
# string.punctuation doesn't consider these different quotation marks by default
PUNCTUATION = string.punctuation + r'“' + r'”'
REFERENCE_SUMMARY_FILES = (
    "high_abstraction.txt",
    "low_abstraction.txt",
    "noise.txt",
)
STOP_WORDS = stopwords.words('english')
# Headers to be ignored in the source extraction
UNWANTED_CHAPTERS = (
    "REFERENCES",
    "LIST OF REFERENCES",
    "CITATIONS",
    "APPENDIX",
    "APPENDIXES",
    "TABLE OF CONTENTS",
    "CONTENTS",
)
WORD_COUNT = 10
