#!/bin/bash
exec gunicorn --bind 0.0.0.0:${PORT:-10000} app_ai:app
