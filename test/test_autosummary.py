import logging
import math
import unittest

import autosummary
import autosummary.autosummary
import autosummary.config

_logger = logging.getLogger()
_DEFAULT_CONFIG = {
    "source_type": "url",
    "source_path": None,
    "word_count": autosummary.config.WORD_COUNT,
    "keyword": None,
    "ne_filter": autosummary.config.NE_FILTER,
    "evaluate-count": 0,
    # TODO: Make configurable
    "evaluate-random": False,
}


class TestSummarizer(unittest.TestCase):

    def test_springer_summary_freq(self):
        config = _DEFAULT_CONFIG.copy()
        config["source_path"] = "https://link.springer.com/chapter/10.1007/978-81-322-3972-7_19"
        config["keyword"] = "freq"
        extracted_summary = autosummary.autosummary.summary_by_config(config)
        assert extracted_summary == "Natural Language Processing | SpringerLink The abundant volume of natural language text in the connected world, though having a large content of knowledge, but it is becoming increasingly difficult to disseminate it by a human to discover the knowledge/wisdom in it, specifically within any given time limits. The automated NLP is aimed to do this job effectively and with accuracy, like a human does it (for a limited of amount text). This chapter presents the challenges of NLP, progress so far made in this field, NLP applications, components of NLP, and grammar of English language—the way machine requires it. In addition, covers the specific areas like probabilistic parsing, ambiguities and their resolution, information extraction, discourse analysis, NL question-answering, commonsense interfaces, commonsense thinking and reasoning, causal-diversity, and various tools for NLP. Give one example of the following ambiguities: a.Phonetic b.Syntactic c.Pragmatic   3. Develop the parse-tree to generate the sentence “Rajan slept on the bench” using following rewrite rules:  5. Draw the tree structures for the following sentences: a.I would like to fly on Air India. Construct the necessary trees. Construct the grammars and parse-tree for the following sentences."

    def test_springer_summary_rake(self):
        config = _DEFAULT_CONFIG.copy()
        config["source_path"] = "https://link.springer.com/chapter/10.1007/978-81-322-3972-7_19"
        config["keyword"] = "rake"
        extracted_summary = autosummary.autosummary.summary_by_config(config)
        assert extracted_summary == "The abundant volume of natural language text in the connected world, though having a large content of knowledge, but it is becoming increasingly difficult to disseminate it by a human to discover the knowledge/wisdom in it, specifically within any given time limits. Draw the tree for the following phrases: a.after 5 pm. b.on Tuesday. Also write the steps. The village was looted by dacoits. The boy who was sleeping was awakened. The boy who was sleeping on the table was awakened. Also, specify whether the ambiguities are syntactic, semantic, or some other? K. R. Chowdhary1Email author1.Department of Computer Science and EngineeringJodhpur Institute of Engineering and TechnologyJodhpurIndia"

    def test_evaluate_one_freq(self):
        config = _DEFAULT_CONFIG.copy()
        config["keyword"] = "freq"
        config["source_path"] = autosummary.config.DATASET_DIRECTORY + autosummary.config.DATASET_FILE
        config["evaluate-count"] = 1
        eval_results = autosummary.autosummary.evaluate_summaries(config)
        summaries, eval_metrics = eval_results[0]
        extracted_summary = summaries[1]
        _logger.debug("RAKE EVAL METRICS: {}".format(eval_metrics))
        assert extracted_summary == "Germanwings crash: Reports of video from plane denied - CNN.com Report: Video captures final moments inside plane Report: Video captures final moments inside planeReport: Video of Germanwings crash existsAirline knew about co-pilot's depressionReport: Ex-girlfriend recalls co-pilot's dark sideAuthorities are not disputing new plane crash detailsAnswers to your Germanwings plane crash questionsGermanwings pilot identifiedVideo shows Andreas Lubitz flying gliderGermanwings pilot's reassuring words go viralCrash recovery worker: Most bodies aren't 'in one piece'Reports: Cell phone video shows Germanwings' final momentsGerman prosecutor: Co-pilot suicidal in the pastHow to prevent the next aircraft tragedyWho was Andreas Lubitz?Investigators: Recovery effort going 'bit by bit'Germanwings victim's family speaks outWhat the Germanwings captain was up againstWhen pilots intentionally crash planesHow did co-pilot keep the captain out of the cockpit?Prosecutor: Deliberate attempt to destroy aircraftProsecutor: Germanwings passengers screamed before crashGermanwings captain: 'For God's sake, open the door'  Marseille prosecutor says \"so far no videos were used in the crash investigation\" despite media reportsJournalists at Bild and Paris Match are \"very confident\" the video clip is real, an editor saysAndreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says Report: Video captures final moments inside plane Report: Video captures final moments inside plane Iran plane buzzed Navy copterVideo shows Germanwings flight's final secondsGov. Germanwings co-pilot reported depression during trainingArchitects plan skyscraper that 'casts no shadow'"
        assert math.isclose(eval_metrics["rouge2-precision"], 0.060446009389671366, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge2-recall"], 0.944954128440367, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge3-precision"], 0.05637110980622431, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge3-recall"], 0.8888888888888888, rel_tol=1e-09, abs_tol=0.0)

    def test_evaluate_one_rake(self):
        config = _DEFAULT_CONFIG.copy()
        config["keyword"] = "rake"
        config["source_path"] = autosummary.config.DATASET_DIRECTORY + autosummary.config.DATASET_FILE
        config["evaluate-count"] = 1
        eval_results = autosummary.autosummary.evaluate_summaries(config)
        summaries, eval_metrics = eval_results[0]
        extracted_summary = summaries[1]
        assert extracted_summary == "Germanwings crash: Reports of video from plane denied - CNN.com Report: Video captures final moments inside plane Report: Video captures final moments inside planeReport: Video of Germanwings crash existsAirline knew about co-pilot's depressionReport: Ex-girlfriend recalls co-pilot's dark sideAuthorities are not disputing new plane crash detailsAnswers to your Germanwings plane crash questionsGermanwings pilot identifiedVideo shows Andreas Lubitz flying gliderGermanwings pilot's reassuring words go viralCrash recovery worker: Most bodies aren't 'in one piece'Reports: Cell phone video shows Germanwings' final momentsGerman prosecutor: Co-pilot suicidal in the pastHow to prevent the next aircraft tragedyWho was Andreas Lubitz?Investigators: Recovery effort going 'bit by bit'Germanwings victim's family speaks outWhat the Germanwings captain was up againstWhen pilots intentionally crash planesHow did co-pilot keep the captain out of the cockpit?Prosecutor: Deliberate attempt to destroy aircraftProsecutor: Germanwings passengers screamed before crashGermanwings captain: 'For God's sake, open the door'  Marseille prosecutor says \"so far no videos were used in the crash investigation\" despite media reportsJournalists at Bild and Paris Match are \"very confident\" the video clip is real, an editor saysAndreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says Germanwings co-pilot reported depression during trainingArchitects plan skyscraper that 'casts no shadow'"
        _logger.debug("RAKE EVAL METRICS: {}".format(eval_metrics))
        assert math.isclose(eval_metrics["rouge2-precision"], 0.06758530183727034, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge2-recall"], 0.944954128440367, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge3-precision"], 0.06303348653972422, rel_tol=1e-09, abs_tol=0.0)
        assert math.isclose(eval_metrics["rouge3-recall"], 0.8888888888888888, rel_tol=1e-09, abs_tol=0.0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format=autosummary.config.LOGGING_FORMAT)
    unittest.main()
