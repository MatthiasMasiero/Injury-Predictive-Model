# Injury Prediction POC

Proof-of-concept for predicting athlete injuries using Catapult Sports sensor data.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key in `.env` (already done)

## Usage

### 1. Collect Data from Catapult API

**Collect a single day:**
```bash
python scripts/data_collector.py --date 2024-08-16
```

**Collect a date range:**
```bash
python scripts/data_collector.py --start-date 2024-08-01 --end-date 2024-08-31
```

Data will be saved to `data/` folder as JSON files.

### 2. Create Injury Labels

(Coming next - will use your injury dates to label sessions)

### 3. Train Model

(Coming next - will train injury prediction model)

## Project Structure

```
injury-prediction-poc/
├── .env                    # API credentials
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/                # Configuration files
├── data/                  # Raw JSON data from API
└── scripts/               # Python scripts
    └── data_collector.py  # Fetch data from Catapult API
```

## Next Steps

1. Collect sample data for testing (10-20 sessions)
2. Create injury labels based on injury dates
3. Engineer rolling window features
4. Train simple prediction model
5. Evaluate on test set
