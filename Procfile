# Managed-platform process definition (Railway / Fly / Heroku-style).
# gunicorn runs the Flask app; wsgi.conf reads env for port/workers.
web: gunicorn wsgi:application -c gunicorn.conf.py
