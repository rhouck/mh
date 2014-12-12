# check for environment variables on server if live/dev mode, otherwise load local environment vars

from os import environ
host = environ.get('MODE', '')
from re import search


if search('live', host):
    pass
elif search('dev', host):
    pass
else:
    import env_vars
