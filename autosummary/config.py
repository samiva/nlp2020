import string

from nltk.corpus import stopwords

DATASET_FILE = "/home/toni/Projects/Uni/nlp2020/files/test_article_links.txt"
LOGGING_FORMAT = "%(asctime)s %(module)s [%(levelname)s]: %(message)s"
# Spacy tags for the named entities we're interested in using for the summarization logic
NAMED_ENTITY_TAGS = ("PERSON", "ORG")
# Specifies whether named entities will be filtered out of keywords.
NE_FILTER = False
# string.punctuation doesn't consider these different quotation marks by default
PUNCTUATION = string.punctuation + r'“' + r'”'
# Note: Remember to set these before using the module!
REFERENCE_SUMMARY_FILES = (
    "../nlp2020/files/high_abstraction.txt",
    "../nlp2020/files/low_abstraction.txt",
    "../nlp2020/files/noise.txt",
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
