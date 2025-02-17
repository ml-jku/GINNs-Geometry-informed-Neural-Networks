import sys
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from petsc4py import PETSc

logger = logging.getLogger("barrier_dolfin")
logger.setLevel(DEBUG)

__all__ = ('log', 'debug', 'info', 'warning', 'error', 'critical',
           'info_red', 'info_green', 'info_blue')

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_red(message, *args, **kwargs):
    ''' Write info message in red.

    :arg message: the message to be printed. '''
    PETSc.Sys.Print(RED % message, *args, **kwargs)


def info_green(message, *args, **kwargs):
    ''' Write info message in green.

    :arg message: the message to be printed. '''
    PETSc.Sys.Print(GREEN % message, *args, **kwargs)


def info_blue(message, *args, **kwargs):
    ''' Write info message in blue.

    :arg message: the message to be printed. '''
    PETSc.Sys.Print(BLUE % message, *args, **kwargs)
