#!/bin/bash

source .venv/bin/activate

export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python main.py
