# Football Video Analysis

## Overview

This project aims to analyze football matches using video data to identify key moments such as the number of players on the field and goal events.

## Directory Structure

FootballData/
│
├── data/
│ ├── videos/
│ │ └── match1.mp4
│ ├── images/
│ │ ├── train/
│ │ │ ├── images/
│ │ │ └── labels/
│ │ └── val/
│ │ ├── images/
│ │ └── labels/
│ └── annotations/
│ └── match1_annotations.json
│
├── notebooks/
│ └── exploratory_data_analysis.ipynb
│
├── src/
│ ├── init .py
│ ├── data/
│ │ ├── init .py
│ │ ├── load_data.py
│ │ ├── preprocess_data.py
│ │ └── goal_dataset.py
│ ├── models/
│ │ ├── init .py
│ │ ├── player_detection.py
│ │ ├── goal_detection.py
│ │ ├── train_goal_detector.py
│ │ └── evaluate_goal_detector.py
│ ├── utils/
│ │ ├── init .py
│ │ ├── visualization.py
│ │ └── video_processing.py
│ └── main.py
│
├── reports/
│ └── video_analysis_report.md
│
├── requirements.txt
└── README.md

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/25PRLLX/FootballData.git
   cd FootballData

2. Install dependencies:

pip install -r requirements.txt

## Usage

1. Run the exploratory data analysis notebook:

jupyter notebook notebooks/exploratory_data_analysis.ipynb

2. Execute the main script:

python src/main.py

## Contributing

Feel free to contribute to the project by opening issues or pull requests.