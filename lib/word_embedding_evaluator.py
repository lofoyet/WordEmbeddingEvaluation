# -*- coding: utf-8 -*-
"""W2V evaluator."""
import os
import json
import uuid
import codecs
import logging
import collections

from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

from textutils import clean_text


# set logging
formatter = logging.Formatter("%(levelname)s - %(message)s")
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(sh)

# set np random seed
TRAIN_RANDOM = np.random.RandomState(7)

FILE_PATH = os.path.abspath(__file__)
CUR_PATH = os.path.dirname(FILE_PATH)
BASE_PATH = os.path.dirname(CUR_PATH)
DATA_PATH = os.path.join(BASE_PATH, "resources")


class BaseEvalClass(object):
    """Base class for all evaluator."""

    eval_folder = ""

    def __init__(self):
        """Init all metrics in dict."""
        self.metrics = {}
        self.tid = str(uuid.uuid1())

    def _init_params(self, params=None):
        """Init for all subinstance."""
        if params:
            for k, v in params.items():
                if k != "self":
                    setattr(self, k, v)

    def prepare(self):
        """Prepare data."""
        raise NotImplementedError

    def evaluate(self):
        """Evaluate scores."""
        raise NotImplementedError

    def save(self):
        """Save to disk."""
        raise NotImplementedError


class WordSimilarity(BaseEvalClass):
    """Similarity evaluation base class, do not initiate."""

    def __init__(self, **kwargs):
        """Init."""
        super(WordSimilarity, self).__init__()
        self._init_params(kwargs)
        self.eval_file = "{}-{}.csv".format(self.seq_id, self.tid)
        self.output_path = \
            os.path.join(self.output_dir, self.eval_folder, self.eval_file)
        self.metrics["mse"] = None
        self.metrics["error"] = None
        self.metrics["size"] = None

    def prepare(self):
        """Prepare WordSimilarity-353 test data."""
        raise NotImplementedError

    def evaluate(self, embedding):
        """Evaluate by calculating correlation scores."""
        mask = embedding[0].isin(self.pairs)
        embedding = embedding[mask].copy().reset_index(drop=True)
        words, embedding_vec = \
            embedding.iloc[:, 0], embedding.iloc[:, 1:].values
        words_to_index = {w: i for i, w in words.iteritems()}
        corr = np.corrcoef(embedding_vec)  # cannot be too large
        self.metrics["error"] = 0
        self.results = [u"word1,word2,human_score,emb_score\n"]
        for word1 in self.pairs:
            for word2 in self.pairs[word1]:
                if word1 in words_to_index and word2 in words_to_index:
                    index1 = words_to_index[word1]
                    index2 = words_to_index[word2]
                    human_score = self.pairs[word1][word2]
                    emb_score = corr[index1][index2]
                    result = "{},{},{},{}\n".format(
                        word1, word2, human_score, emb_score)
                    self.results.append(result)
                    self.metrics["error"] += np.square(human_score - emb_score)
        self.metrics["mse"] = \
            self.metrics["error"] / len(self.results)  # double count
        self.metrics["size"] = len(self.results) / 2
        self.metrics["path"] = self.output_path

    def save(self):
        """Save to csv file."""
        with codecs.open(self.output_path, "w", encoding="utf-8") as f:
            f.write("".join(self.results))


class WordSimilarity353(WordSimilarity):
    """Similarity evaluation."""

    file_params = {
        "word_similarity_file":
            "WordSimilarity-353.csv",
        "word_similarity_zipped_file":
            "combined.csv",
        "eval_folder":
            "WordSimilarity353",
        "web_source":
            "http://www.cs.technion.ac.il/~gabr/resources/data"
            "/wordsim353/wordsim353.zip"
    }

    def __init__(self, **kwargs):
        """Init."""
        kwargs.update(self.file_params)
        super(WordSimilarity353, self).__init__(**kwargs)

    def prepare(self):
        """Prepare WordSimilarity-353 test data."""
        data_path = os.path.join(
            DATA_PATH,
            self.word_similarity_file
        )
        if os.path.isfile(data_path):  # read local
            eval_data = pd.read_csv(data_path, encoding="utf-8")
        else:  # read web
            resp = urlopen(self.web_source)
            zipfile = ZipFile(StringIO(resp.read()))
            raw_data = StringIO()
            word_file = zipfile.open(self.word_similarity_zipped_file)
            for line in word_file.readlines():
                raw_data.write(line)
            raw_data.seek(0)
            word_file.close()
            eval_data = pd.read_csv(raw_data, encoding="utf-8")
        self.pairs = {}
        # symmetric score
        for index, row in eval_data.iterrows():
            word1, word2, score = \
                row["Word 1"], row["Word 2"], row["Human (mean)"]
            if word1 not in self.pairs:
                self.pairs[word1] = {}
            else:
                self.pairs[word1][word2] = score / 10  # normalize to 0-1
            if word2 not in self.pairs:
                self.pairs[word2] = {}
            else:
                self.pairs[word2][word1] = score / 10


class TfidfEmbeddingVectorizer(BaseEstimator):
    """Weigt word embedding by tf-idf and average."""

    def __init__(self, word2index, embedding, unk_index=0):
        """Init."""
        self.word2index = word2index
        self.embedding = embedding
        self.unk_index = unk_index
        self.dim = self.embedding.shape[1]

    def fit(self, train_docs):
        """Fit model."""
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(train_docs)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,  # defaul weight
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    def transform(self, docs):
        """Transform any list of docs to embedding."""
        doc_vectors = []
        for words in docs:
            word_vectors = []
            for w in words:
                index = self.word2index.get(w, self.unk_index)
                vector = self.embedding[index]
                weight = self.word2weight[w]
                weighted_vector = weight * vector
                word_vectors.append(weighted_vector)
            if word_vectors:
                word_vectors = np.mean(word_vectors, axis=0)
            else:
                word_vectors = np.zeros(self.dim)
            doc_vectors.append(word_vectors)
        doc_vectors = np.array(doc_vectors)
        return doc_vectors


class DocumentClassifer(BaseEvalClass):
    """
    Classification evaluation on pretrained embedding.

    This is a parent class that should not be direclty used.
    Use all the subclass of this.
    """

    def __init__(self, **kwargs):
        """Init."""
        super(DocumentClassifer, self).__init__()
        self._init_params(kwargs)
        self.eval_file = "{}-{}.csv".format(self.task_id, self.uuid)
        self.output_path = \
            os.path.join(self.output_dir, self.eval_folder, self.eval_file)

        self.metrics["confusion_matrix"] = None
        self.metrics["accuracy"] = None
        self.metrics["recall"] = None
        self.metrics["precision"] = None
        self.metrics["train_size"] = None
        self.metrics["test_size"] = None

    def train(self, word2index, embedding, unk_index):
        """Train a multi classificaion."""
        # use tfidf embedding model to generate doc embedding
        self.vectorizer = TfidfEmbeddingVectorizer(
            word2index, embedding, unk_index)
        train_x, train_y = zip(*self.train_data)
        test_x, test_y = zip(*self.test_data)
        # fit train data only to evaluate ability of predicting
        self.vectorizer.fit(train_x)
        train_x = self.vectorizer.transform(train_x)
        test_x = self.vectorizer.transform(test_x)
        # normalize to same scale
        self.model = MLPClassifier(
            learning_rate="invscaling",
            max_iter=1000)
        self.model.fit(train_x, train_y)
        pred_test_y = self.model.predict(test_x)
        return test_y, pred_test_y

    def evaluate(self, embedding):
        """Evaluate based on classification metrics."""
        words, embedding_vec = \
            embedding.iloc[:, 0], embedding.iloc[:, 1:].values
        word2index = {w: i for i, w in words.iteritems()}
        unk_index = 0
        test_y, pred_test_y = \
            self.train(word2index, embedding_vec, unk_index)
        test_cf = confusion_matrix(test_y, pred_test_y)
        test_accuracy = accuracy_score(test_y, pred_test_y)
        test_precision = precision_score(test_y, pred_test_y, average=None)
        test_recall = recall_score(test_y, pred_test_y, average=None)
        self.metrics["confusion_matrix"] = test_cf.tolist()
        self.metrics["accuracy"] = test_accuracy.tolist()
        self.metrics["recall"] = test_recall.tolist()
        self.metrics["precision"] = test_precision.tolist()

    def save(self):
        """Save predictions."""
        pass


class NewsGroups20Classifer20(DocumentClassifer):
    """Classification evaluation on pretrained embedding."""

    file_params = {
        "classes":
            [
                "comp.sys.mac.hardware",
                "rec.sport.baseball",
                "sci.med",
                "soc.religion.christian",
                "talk.politics.guns"
            ],
        "train_ratio": 0.6,
        "eval_folder":
            "NewsGroups20Classifer20",
        "web_source":
            "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz"
    }

    def __init__(self, **kwargs):
        """Init."""
        kwargs.update(self.file_params)
        super(NewsGroups20Classifer20, self).__init__(**kwargs)

    def prepare(self):
        """Prepare training data."""
        # first tokenize
        tokenizer = TweetTokenizer()
        data_base_dir = os.path.join(DATA_PATH, "20news-18828")
        self.data = []
        for label, class_ in enumerate(self.classes):  # find class article
            class_dir = os.path.join(data_base_dir, class_)
            for file_ in os.listdir(class_dir):
                if not file_.isdigit():  # valid files are digit like
                    continue
                file_path = os.path.join(class_dir, file_)
                with open(file_path, "rb") as f:
                    texts = f.readlines()[2:]  # strip first two header lines
                texts = [
                    clean_text(text.decode("utf-8", errors="ignore"))
                    for text in texts
                ]
                tokens = []
                for text in texts:
                    tokens += tokenizer.tokenize(text)
                self.data.append([tokens, label])
        # split into train and test
        TRAIN_RANDOM.shuffle(self.data)
        total_len = len(self.data)
        self.metrics["train_size"] = train_size = int(total_len * 0.6)
        self.metrics["test_size"] = total_len - train_size
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]


class Scheduler(object):
    """The class to evaluate word embedding."""

    _defined_tasks = [
        ["word_similarity_353", WordSimilarity353]
        ["doc_classification_20_news",
            NewsGroups20Classifer20],
    ]
    defined_tasks = collections.OrderedDict()
    for task_name, task in _defined_tasks:
        defined_tasks[task_name] = task
    log_file = "eval_history.json"

    def __init__(self, output_dir, w2v_path, sep, quoting, embedding=None):
        """Init."""
        self.w2v_path = w2v_path
        self.sep = sep
        self.quoting = quoting
        self.embedding = embedding
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, self.log_file)
        self.log_dict = {"output_dir": output_dir}

    def _save_log(self):
        """Save all log."""
        with open(self.log_path, "aw") as f:
            f.write("{}\n".format(json.dumps(self.log_dict)))

    def _prepare_embedding(self):
        """Read w2v embedding into class."""
        if self.embedding is None:
            self.embedding = \
                pd.read_csv(
                    self.w2v_path,
                    sep=self.sep,
                    header=None,
                    encoding="utf-8",
                    quoting=self.quoting)

    def add_tasks(self, *args):
        """Add tasks to do."""
        self.registered_tasks = collections.OrderedDict()
        for seq_id, (task_name, task) in \
                enumerate(self.defined_tasks.iteritems()):
            if task_name in args:
                kwargs = {
                    "output_dir": self.output_dir,
                    "seq_id": seq_id
                }
                self.registered_tasks[task_name] = task(**kwargs)

    def run(self):
        """Prepare evalution for all tasks."""
        self._prepare_embedding()
        for task_name, task in self.registered_tasks.iteritems():
            task.prepare()
            task.evaluate(self.embedding)
            task.save()
            logger.info("Task {} finished.".format(task_name))
            self.log_dict[task_name] = task.metrics
        self._save_log()
