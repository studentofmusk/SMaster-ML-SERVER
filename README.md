# SMaster ML Server - Real-Time Sign Language Detection

This Flask server uses Mediapipe and an LSTM model to detect sign language actions from real-time video inputs. It serves predictions to the mobile app.

## Features
- Accepts real-time landmark sequences
- Processes data using an LSTM model
- Returns predicted sign language actions
- Handles pre-processing with Mediapipe

## Prerequisites
- Python 3.8+
- pip

## Setup Instructions

```sh
git clone https://github.com/studentofmusk/SMaster-ML-SERVER
cd SMaster-ML-SERVER
pip install -r requirements.txt
python app.py
