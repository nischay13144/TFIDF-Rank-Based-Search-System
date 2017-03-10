import re
import unicodedata
import logging
from util import timed
from search import Indexable
from search import SearchEngine

logger = logging.getLogger(__name__)

class Book(Indexable):

    def __init__(self, iid, title, author, metadata):
        Indexable.__init__(self, iid, metadata)
        self.title = title
        self.author = author

    def __repr__(self):
        
        return 'question: %s, answer: %s, document_Name: %s' % \
               (self.title, self.author, self.iid)


class BookDataPreprocessor(object):

    _EXTRA_SPACE_REGEX = re.compile(r'\s+', re.IGNORECASE)
    _SPECIAL_CHAR_REGEX = re.compile(
        r"(?P<p>(\.+)|(\?+)|(!+)|(:+)|(;+)|"
        r"(\(+)|(\)+)|(\}+)|(\{+)|('+)|(-+)|(\[+)|(\]+)|"
        r"(?<!\d)(,+)(?!=\d)|(\$+))")

    def preprocess(self, entry):

        f_entry = entry.lower()
        f_entry = f_entry.replace('\t', '|').strip()

        f_entry = self.strip_accents(unicode(f_entry, 'ISO-8859-1'))
        f_entry = self._SPECIAL_CHAR_REGEX.sub(' ', f_entry)
        f_entry = self._EXTRA_SPACE_REGEX.sub(' ', f_entry)

        book_desc = f_entry.split('|')

        return book_desc

    def strip_accents(self, text):
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore')


class BookInventory(object):

    _BOOK_META_ID_INDEX = 0
    _BOOK_META_TITLE_INDEX = 1 #question
    _BOOK_META_AUTHOR_INDEX = 2 #answer
    _NO_RESULTS_MESSAGE = 'Sorry, no results.'

    def __init__(self, filename):
        self.filename = filename
        self.engine = SearchEngine()

    @timed
    def load_books(self):
        processor = BookDataPreprocessor()
        with open(self.filename) as catalog:
            for entry in catalog:
                book_desc = processor.preprocess(entry)
                metadata = ' '.join(book_desc[self._BOOK_META_TITLE_INDEX:])

                iid = book_desc[self._BOOK_META_ID_INDEX].strip()
                title = book_desc[self._BOOK_META_TITLE_INDEX].strip()
                author = book_desc[self._BOOK_META_AUTHOR_INDEX].strip()

                book = Book(iid, title, author, metadata)
                self.engine.add_object(book)

        self.engine.start()

    @timed
    def search_books(self, query, n_results=10):

        result = ''
        if len(query) > 0:
            result = self.engine.search(query, n_results)

        if len(result) > 0:
            return '\n'.join([str(indexable) for indexable in result])
        return self._NO_RESULTS_MESSAGE

    def books_count(self):
        return self.engine.count()

