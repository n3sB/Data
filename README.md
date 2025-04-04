# Heart Disease Prediction Analysis using BRFSS 2015 Dataset

This project analyzes the BRFSS (Behavioral Risk Factor Surveillance System) 2015 dataset to develop predictive models for heart disease. The analysis includes data exploration, preprocessing, and the development of machine learning models to predict heart disease occurrence.

## Project Structure

- `heart_disease_analysis.py`: Main analysis script
- `requirements.txt`: Required Python packages
- `eda_plots.png`: Generated exploratory data analysis plots
- `model_results.txt`: Model evaluation metrics

## Setup and Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Analysis

1. Ensure the BRFSS2015.csv file is in the project directory
2. Run the analysis:

```bash
python heart_disease_analysis.py
```

## Analysis Components

1. Data Exploration & Preprocessing

   - Dataset overview and missing values analysis
   - Exploratory Data Analysis (EDA)
   - Data preprocessing (encoding, scaling, handling missing values)

2. Model Development & Evaluation

   - Class imbalance handling using SMOTE
   - Multiple classification models (Logistic Regression, Random Forest, SVM)
   - Model evaluation using various metrics

3. Results
   - Model performance metrics are saved in `model_results.txt`
   - Visualizations are saved in `eda_plots.png`

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies
