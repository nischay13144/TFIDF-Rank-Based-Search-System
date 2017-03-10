import numpy as np
import scipy.sparse as sp
import scipy.sparse.sparsetools as sptools
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

STOP_WORDS_FILENAME = 'data/stop_words.txt'

class Indexable(object):

    def __init__(self, iid, metadata):
        self.iid = iid
        self.words_count = defaultdict(int)

        for word in metadata.split():
            self.words_count[word] += 1

    def __repr__(self):
        return ' '.join(self.words_count.keys()[:10])

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def words_generator(self, stop_words):

        for word in self.words_count.keys():
            if word not in stop_words or len(word) > 5:
                yield word

    def count_for_word(self, word):
        return self.words_count[word] if word in self.words_count else 0


class IndexableResult(object):

    def __init__(self, score, indexable):
        self.score = score
        self.indexable = indexable

    def __repr__(self):
        return '\nRanking score: %f,  %s' % (self.score, self.indexable)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and abs(self.score - other.score) < 0.0001
                and self.indexable == other.indexable)

    def __ne__(self, other):
        return not self.__eq__(other)


class TfidfRank(object):

    def __init__(self, stop_words, smoothing=1):
        self.smoothing = smoothing
        self.stop_words = stop_words
        self.vocabulary = {}
        self.ft_matrix = []
        self.ifd_diag_matrix = []
        self.tf_idf_matrix = []

    def build_rank(self, objects):

        self.__build_vocabulary(objects)

        n_terms = len(self.vocabulary)
        n_docs = len(objects)
        ft_matrix = sp.lil_matrix((n_docs, n_terms), dtype=np.dtype(float))

        
        for index, indexable in enumerate(objects):
            for word in indexable.words_generator(self.stop_words):
                word_index_in_vocabulary = self.vocabulary[word]
                doc_word_count = indexable.count_for_word(word)
                ft_matrix[index, word_index_in_vocabulary] = doc_word_count
        self.ft_matrix = ft_matrix.tocsc()

        logger.info('Results will be displayed from higher to lower ranking...')

        df = np.diff(self.ft_matrix.indptr) + self.smoothing
        n_docs_smooth = n_docs + self.smoothing

        idf = np.log(float(n_docs_smooth) / df) + 1.0
        self.ifd_diag_matrix = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)


        self.tf_idf_matrix = self.ft_matrix * self.ifd_diag_matrix
        self.tf_idf_matrix = self.tf_idf_matrix.tocsr()

        norm = self.tf_idf_matrix.tocsr(copy=True)
        norm.data **= 2
        norm = norm.sum(axis=1)
        n_nzeros = np.where(norm > 0)
        norm[n_nzeros] = 1.0 / np.sqrt(norm[n_nzeros])
        norm = np.array(norm).T[0]
        sptools.csr_scale_rows(self.tf_idf_matrix.shape[0],
                                      self.tf_idf_matrix.shape[1],
                                      self.tf_idf_matrix.indptr,
                                      self.tf_idf_matrix.indices,
                                      self.tf_idf_matrix.data, norm)

    def __build_vocabulary(self, objects):

        vocabulary_index = 0
        for indexable in objects:
            for word in indexable.words_generator(self.stop_words):
                if word not in self.vocabulary:
                    self.vocabulary[word] = vocabulary_index
                    vocabulary_index += 1

    def compute_rank(self, doc_index, terms):

        score = 0
        for term in terms:
            term_index = self.vocabulary[term]
            score += self.tf_idf_matrix[doc_index, term_index]
        return score


class Index(object):

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.term_index = defaultdict(list)

    def build_index(self, objects):
        
        for position, indexable in enumerate(objects):
            for word in indexable.words_generator(self.stop_words):
                self.term_index[word].append(position)

    def search_terms(self, terms):

        docs_indices = []
        for term_index, term in enumerate(terms):

            # Here I keep only docs that contain all terms
            if term not in self.term_index:
                docs_indices = []
                break

            
            docs_with_term = self.term_index[term]
            if term_index == 0:
                docs_indices = docs_with_term
            else:
                docs_indices = set(docs_indices) & set(docs_with_term)
        return list(docs_indices)


class SearchEngine(object):

    def __init__(self):
        self.objects = []
        self.stop_words = self.__load_stop_words()
        self.rank = TfidfRank(self.stop_words)
        self.index = Index(self.stop_words)

    def __load_stop_words(self):
        stop_words = {}
        with open(STOP_WORDS_FILENAME) as stop_words_file:
            for word in stop_words_file:
                stop_words[word.strip()] = True
        return stop_words

    def add_object(self, indexable):
        self.objects.append(indexable)

    def start(self):
        logger.info('Starting search engine...')
        self.index.build_index(self.objects)
        self.rank.build_rank(self.objects)

    def search(self, query, n_results=10):
        terms = query.lower().split()
        docs_indices = self.index.search_terms(terms)
        search_results = []

        for doc_index in docs_indices:
            indexable = self.objects[doc_index]
            doc_score = self.rank.compute_rank(doc_index, terms)
            result = IndexableResult(doc_score, indexable)
            search_results.append(result)

        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:n_results]

    def count(self):
        return len(self.objects)

