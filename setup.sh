#!/bin/bash
# Run it:

apt update && apt install git python3-pip python3-venv
git clone -b igeon https://github.com/pantierra/efast.git && cd efast
python3 -m venv .venv && source .venv/bin/activate && pip install -e .[dev]
export TERM=xterm-256color && screen
export $(grep -v '^#' .env | xargs) && python run_efast_site.py --sitename innsbruck --season 2021 --log-level INFO
