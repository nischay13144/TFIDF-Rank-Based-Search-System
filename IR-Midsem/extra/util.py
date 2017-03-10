import time
import logging

logger = logging.getLogger(__name__)

def timed(fn):

    def wrapped(*arg, **kw):
        ts = time.time()
        result = fn(*arg, **kw)
        te = time.time()
        return result

    return wrapped