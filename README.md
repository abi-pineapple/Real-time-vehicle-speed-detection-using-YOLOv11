# Vehicle Speed Detector

Real-time vehicle speed detection using YOLOv11 and Streamlit.

## Features
-  Real-time vehicle detection
-  Speed calculation from motion
-  Speeding violation detection
-  CSV export of violations
-  Web interface

## Installation

```bash
pip install -r requirements.txt
```

## Setup

```bash
cp .env.example .env
# Add your Mistral API key to .env
```

## Usage

```bash
streamlit run app.py
```

Upload a video and configure the speed limit. The system will detect vehicles and flag speeders.

## How It Works

1. Detects vehicles using YOLOv11
2. Calculates speed from pixel displacement
3. Flags violations if speed > limit
4. Exports results to CSV

## Requirements

- Python 3.10+
- YOLOv11
- OpenCV
- Streamlit
- Mistral AI API key

---

**Built with YOLOv11 • OpenCV • Streamlit**