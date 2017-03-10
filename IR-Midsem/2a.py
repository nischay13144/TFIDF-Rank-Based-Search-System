import sys
import optparse
import logging
sys.path.append('extra')
import book_a

DEBUG = True

log_level = logging.DEBUG if DEBUG else logging.INFO
log_format = '%(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

CATALOG_FILENAME = 'data/question_answer_pairs.txt'


def execute_search(data_location):
    query = None
    repository = book_a.BookInventory(data_location)
    logger.info('Ranking on the basis of answers')
    logger.info('Loading dataset...')

    repository.load_books()
    docs_number = repository.books_count()
    logger.info('Done loading dataset, %d docs in index', docs_number)

    while query is not '':

        query = raw_input('\nEnter a query (dont use ? at the end): ')
        search_results = repository.search_books(query)

        print search_results


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data',
                      dest='data',
                      help='Location of the data that will be indexed',
                      default=CATALOG_FILENAME)

    options, args = parser.parse_args()
    execute_search(options.data)
