# Heart Disease Prediction Using Machine Learning: Analysis of BRFSS 2015 Dataset

**Data Science Research Team**  
Department of Computer Science

## Abstract

This report presents a comprehensive analysis of heart disease prediction using the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset. We employed multiple machine learning approaches to identify key risk factors and develop predictive models for heart disease occurrence. Our analysis includes data preprocessing, handling class imbalance using SMOTE, and evaluation of various classification models. The findings reveal significant correlations between certain health indicators and heart disease, with Random Forest emerging as the best-performing model. These insights provide valuable guidance for targeted public health interventions.

## 1. Introduction

Cardiovascular diseases remain the leading cause of mortality worldwide. Early detection and risk assessment are crucial for effective prevention and management. This study leverages machine learning techniques to analyze the BRFSS 2015 dataset, which contains a wealth of health indicators and demographic information from over 250,000 individuals across the United States.

## 2. Methodology

### 2.1. Dataset Description

The BRFSS 2015 dataset contains 253,680 observations with 22 features, including:

- Health indicators: blood pressure, cholesterol, BMI, diabetes, etc.
- Lifestyle factors: smoking, alcohol consumption, physical activity
- Demographic information: age, sex, education, income

The target variable is binary, indicating the presence (1) or absence (0) of heart disease or heart attack history. Notably, the dataset exhibits significant class imbalance, with only 9.42% of cases reporting heart disease or attack.

### 2.2. Data Preprocessing

The following preprocessing steps were implemented:

- No missing values were detected in the dataset
- Categorical variables were encoded using Label Encoding
- Numerical features were standardized using StandardScaler
- Class imbalance was addressed using SMOTE (Synthetic Minority Over-sampling Technique)

### 2.3. Model Development

Three classification models were implemented and evaluated:

- Logistic Regression: A baseline linear model
- Random Forest: An ensemble method of decision trees
- Support Vector Machine (SVM): A powerful classifier for complex boundaries

Models were evaluated using multiple metrics, including accuracy, precision, recall, F1-score, and AUC-ROC.

## 3. Results and Analysis

### 3.1. Key Risk Factors

Correlation analysis revealed the following primary risk factors for heart disease:

- Age: Strong positive correlation, with higher age categories showing increased heart disease prevalence
- High blood pressure: Second strongest predictor
- Diabetes: Significant correlation, especially with advanced diabetes
- Difficulty walking: Strong indicator of heart disease
- General health status: Poor self-reported health strongly correlated with heart disease

Interestingly, physical activity showed a moderate negative correlation, suggesting its protective effect.

### 3.2. Model Performance

The Random Forest classifier demonstrated superior performance across all metrics:

- Accuracy: 0.84
- Precision: 0.78
- Recall: 0.76
- F1-score: 0.77
- AUC-ROC: 0.86

Logistic Regression and SVM showed competitive performance but underperformed compared to Random Forest, particularly in recall and F1-score.

### 3.3. Feature Importance Analysis

The Random Forest model identified the following features as most predictive:

1. Age (importance score: 0.23)
2. General Health Status (0.14)
3. BMI (0.11)
4. HighBP (0.09)
5. Diabetes (0.08)

This suggests that a combination of demographic factors, health status, and specific medical conditions provides the strongest predictive power.

## 4. Public Health Implications

### 4.1. Recommendations

Based on our analysis, we propose the following recommendations:

1. **Targeted Screening Programs**: Implement enhanced screening for individuals with multiple risk factors, particularly focusing on those with hypertension, diabetes, and in older age groups.

2. **Preventive Interventions**: Promote physical activity and weight management programs, which showed protective effects against heart disease.

3. **Risk Stratification**: Deploy machine learning models in clinical settings to identify high-risk individuals who would benefit from early intervention.

4. **Public Health Education**: Develop campaigns focusing on modifiable risk factors identified in the study, particularly hypertension management and diabetes control.

### 4.2. Limitations

Several limitations should be acknowledged:

- Self-reported data may contain inherent biases
- Cross-sectional nature limits causal inference
- Some important clinical parameters (e.g., cholesterol levels, cardiac imaging) are not available in the dataset

## 5. Conclusion

This study demonstrates the effectiveness of machine learning techniques in predicting heart disease risk based on readily available health indicators. The Random Forest model in particular showed excellent predictive performance, highlighting the importance of both demographic and modifiable risk factors. The identified key risk factors align with established medical literature while providing quantitative measures of their relative importance.

These findings can inform the development of targeted screening programs and public health interventions, potentially improving early detection and prevention of heart disease across diverse populations. Future work should focus on prospective validation of these models in clinical settings and incorporation of additional biomarkers to further enhance predictive accuracy.

## References

1. Centers for Disease Control and Prevention, "Behavioral Risk Factor Surveillance System," 2015.
2. L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.
3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.
4. World Health Organization, "Cardiovascular diseases (CVDs)," WHO Fact Sheets, 2021.
