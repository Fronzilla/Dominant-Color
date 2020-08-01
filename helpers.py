# -*- coding: utf-8 -*-
""""
The helper module
"""

import warnings
import logging
import sys


# exceptions
class ImageValidation(Exception):
    pass


# logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

# Ignore any warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def ignore_warn(*args, **kwargs):
    pass


# decorators
def accepts():
    def check_accepts(f):
        def new_f(*args, **kwargs):
            if not isinstance(args[1], str):
                raise ImageValidation('Input parameter is not a string')
            if not args[1].endswith('jpg') and not args[1].endswith('png'):
                raise ImageValidation('Only jpg and png images are allowed')

            return f(*args, **kwargs)
        return new_f

    return check_accepts
