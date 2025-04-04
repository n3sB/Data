import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(file_path):
    """
    Load and provide initial exploration of the dataset
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Basic dataset information
    print("\nDataset Overview:")
    print("-" * 50)
    print(f"Number of observations: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Print column names
    print("\nColumn Names:")
    print("-" * 50)
    print(df.columns.tolist())
    
    # Missing values analysis
    print("\nMissing Values Analysis:")
    print("-" * 50)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(df.describe())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Distribution of target variable
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='HeartDiseaseorAttack')
    plt.title('Distribution of Heart Disease Cases')
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png')
    plt.close()

def preprocess_data(df):
    """
    Preprocess the data for modeling
    """
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Ensure target variable is categorical
    df_processed['HeartDiseaseorAttack'] = df_processed['HeartDiseaseorAttack'].astype(int)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numerical_columns] = imputer.fit_transform(df_processed[numerical_columns])
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Scale numerical features (excluding target variable)
    scaler = StandardScaler()
    numerical_columns = [col for col in numerical_columns if col != 'HeartDiseaseorAttack']
    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
    
    return df_processed

def handle_class_imbalance(X, y):
    """
    Apply SMOTE to handle class imbalance
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
        
        print(f"\n{name} Results:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
    
    return results, models

def main():
    # Load and explore data
    df = load_and_explore_data('heart_disease_health_indicators_BRFSS2015.csv')
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('HeartDiseaseorAttack', axis=1)
    y = df_processed['HeartDiseaseorAttack']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Train and evaluate models
    results, models = train_and_evaluate_models(X_train_balanced, X_test, y_train_balanced, y_test)
    
    # Save results
    with open('model_results.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name} Results:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

if __name__ == "__main__":
    main() 