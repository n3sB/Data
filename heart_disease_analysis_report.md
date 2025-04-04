# Predictive Modeling of Heart Disease Using BRFSS 2015 Dataset

**Data Science Research Team**  
Department of Computer Science  
University  
research@university.edu

## Abstract

This study presents a comprehensive analysis of heart disease prediction using the BRFSS 2015 dataset. We employed multiple machine learning approaches to identify key risk factors and develop predictive models for heart disease occurrence. The analysis includes data preprocessing, handling class imbalance, and evaluation of various classification models. Our findings indicate significant correlations between certain health indicators and heart disease, providing valuable insights for public health interventions.

## 1. Introduction

Heart disease remains a leading cause of mortality worldwide. Early detection and risk assessment are crucial for prevention and treatment. This study leverages machine learning techniques to analyze the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset, aiming to develop accurate predictive models for heart disease occurrence.

## 2. Methodology

### 2.1 Dataset Description

The BRFSS 2015 dataset contains 253,680 observations with 22 features, including health indicators, demographic information, and lifestyle factors. The target variable is binary, indicating the presence or absence of heart disease or heart attack history.

### 2.2 Data Preprocessing

Key preprocessing steps included:

- Handling missing values using median imputation
- Encoding categorical variables
- Scaling numerical features using StandardScaler
- Addressing class imbalance using SMOTE

### 2.3 Model Development

We implemented three classification models:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## 3. Results and Discussion

### 3.1 Data Exploration Findings

- The dataset shows a significant class imbalance, with approximately 9.4% of cases positive for heart disease
- Key risk factors identified through correlation analysis include:
  - High blood pressure
  - High cholesterol
  - Diabetes
  - Age

### 3.2 Model Performance

The models were evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

## 4. Public Health Implications

### 4.1 Key Recommendations

1. Implement targeted screening programs for individuals with multiple risk factors
2. Focus on preventive measures for modifiable risk factors
3. Develop early intervention strategies based on predictive modeling

### 4.2 Limitations

- Self-reported data may contain inherent biases
- Cross-sectional nature of the study limits causal inference
- Some important clinical variables may not be captured in the dataset

## 5. Future Work

- Incorporate longitudinal data to track disease progression
- Explore deep learning approaches for improved prediction
- Develop interpretable models for clinical decision support

## 6. Conclusion

This study demonstrates the potential of machine learning in heart disease prediction using population health data. The developed models show promising results in identifying high-risk individuals, though further validation in clinical settings is recommended.

## References

1. Centers for Disease Control and Prevention, "Behavioral Risk Factor Surveillance System," 2015.
2. T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.
3. World Health Organization, "Cardiovascular diseases (CVDs)," WHO Fact Sheets, 2021.
