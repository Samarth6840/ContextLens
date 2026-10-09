# Managed-platform process definition (Railway / Fly / Heroku-style).
# gunicorn runs the Flask app; gunicorn.conf.py reads env for port/workers
# (PORT is honored first; CONTEXTLENS_PORT is the local override).
web: gunicorn wsgi:application -c gunicorn.conf.py
