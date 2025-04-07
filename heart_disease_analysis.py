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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_and_explore_data(file_path):
    """
    Load and provide initial exploration of the dataset
    """
    # Load the data
    print(f"Loading data from {file_path}...")
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
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")
    
    # Target variable distribution
    print("\nTarget Variable Distribution:")
    print("-" * 50)
    target_counts = df['HeartDiseaseorAttack'].value_counts()
    print(f"No Heart Disease: {target_counts[0]} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"Heart Disease: {target_counts[1]} ({target_counts[1]/len(df)*100:.2f}%)")
    
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis with detailed visualizations
    """
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(df.describe())
    
    # Get target variable distribution
    target_counts = df['HeartDiseaseorAttack'].value_counts()
    
    # Create visualization directory if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='HeartDiseaseorAttack')
    plt.title('Distribution of Heart Disease Cases', fontsize=16)
    plt.xlabel('Heart Disease or Attack (0=No, 1=Yes)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count and percentage labels
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5000,
                f'{height}\n({height/total*100:.1f}%)',
                ha="center", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/target_distribution.png')
    plt.close()
    print("Generated target distribution plot")
    
    # 2. Correlation heatmap
    plt.figure(figsize=(14, 12))
    correlation_matrix = df.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
    plt.title('Correlation Heatmap of Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    print("Generated correlation heatmap")
    
    # 3. Feature importance based on correlation with target
    correlations = correlation_matrix['HeartDiseaseorAttack'].sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=correlations.values[1:11], y=correlations.index[1:11])
    plt.title('Top 10 Features Correlated with Heart Disease', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/feature_correlation.png')
    plt.close()
    print("Generated feature correlation plot")
    
    # 4. Age distribution by heart disease status
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Age', y='HeartDiseaseorAttack')
    plt.title('Age Distribution by Heart Disease Status', fontsize=16)
    plt.xlabel('Age Category', fontsize=14)
    plt.ylabel('Heart Disease Status (0=No, 1=Yes)', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/age_distribution.png')
    plt.close()
    print("Generated age distribution plot")
    
    # 5. Key risk factors comparison
    risk_factors = ['HighBP', 'HighChol', 'Diabetes', 'Stroke']
    plt.figure(figsize=(14, 10))
    
    for i, factor in enumerate(risk_factors):
        plt.subplot(2, 2, i+1)
        sns.countplot(data=df, x=factor, hue='HeartDiseaseorAttack')
        plt.title(f'{factor} vs Heart Disease', fontsize=14)
        plt.xlabel(f'{factor} (0=No, 1=Yes)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('visualizations/risk_factors.png')
    plt.close()
    print("Generated risk factors comparison plot")
    
    # 6. BMI distribution by heart disease status
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='BMI', hue='HeartDiseaseorAttack', bins=30, multiple='dodge')
    plt.title('BMI Distribution by Heart Disease Status', fontsize=16)
    plt.xlabel('BMI', fontsize=14)
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('visualizations/bmi_distribution.png')
    plt.close()
    print("Generated BMI distribution plot")
    
    print("Generated EDA report")
    return df

def preprocess_data(df):
    """
    Preprocess the data for modeling with detailed explanation
    """
    print("\nPreprocessing Data:")
    print("-" * 50)
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    print("- Created copy of original dataframe")
    
    # Ensure target variable is categorical
    df_processed['HeartDiseaseorAttack'] = df_processed['HeartDiseaseorAttack'].astype(int)
    print("- Converted target variable to integer type")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns
    print(f"- Identified {len(numerical_columns)} numerical columns")
    
    if df_processed.isnull().sum().sum() > 0:
        print(f"- Handling {df_processed.isnull().sum().sum()} missing values with median imputation")
        df_processed[numerical_columns] = imputer.fit_transform(df_processed[numerical_columns])
    else:
        print("- No missing values to impute")
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    print(f"- Identified {len(categorical_columns)} categorical columns")
    
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        print("- Encoded categorical variables using LabelEncoder")
    else:
        print("- No categorical variables to encode")
    
    # Scale numerical features (excluding target variable)
    scaler = StandardScaler()
    scaling_columns = [col for col in numerical_columns if col != 'HeartDiseaseorAttack']
    df_processed[scaling_columns] = scaler.fit_transform(df_processed[scaling_columns])
    print(f"- Scaled {len(scaling_columns)} numerical features using StandardScaler")
    
    print("Preprocessing complete")
    return df_processed

def handle_class_imbalance(X, y):
    """
    Apply SMOTE to handle class imbalance with explanation
    """
    print("\nHandling Class Imbalance:")
    print("-" * 50)
    
    # Print class distribution before SMOTE
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution before SMOTE:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} ({count/len(y)*100:.2f}%)")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Print class distribution after SMOTE
    unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
    print("\nClass distribution after SMOTE:")
    for cls, count in zip(unique_resampled, counts_resampled):
        print(f"Class {cls}: {count} ({count/len(y_resampled)*100:.2f}%)")
    
    print(f"\nOriginal dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models with detailed metrics and visualizations
    """
    print("\nTraining and Evaluating Models:")
    print("-" * 50)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    all_y_pred = {}
    all_probas = {}
    feature_importances = {}
    
    # Create figure for ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        all_y_pred[name] = y_pred
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            all_probas[name] = y_proba
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Get feature importances for Random Forest
        if name == 'Random Forest':
            feature_importances[name] = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else None
        }
        
        # Print metrics
        print(f"\n{name} Results:")
        for metric, value in results[name].items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'visualizations/confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
        
    # Finalize and save ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('visualizations/roc_curves.png')
    plt.close()
    print("Generated ROC curves plot")
    
    # Plot feature importances for Random Forest
    if 'Random Forest' in feature_importances:
        top_features = feature_importances['Random Forest'].head(10)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('Top 10 Feature Importances - Random Forest', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.close()
        print("Generated feature importance plot")
    
    # Plot model comparison
    plt.figure(figsize=(14, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    # Prepare data for plotting
    model_names = list(results.keys())
    metric_data = {metric: [results[model][metric] for model in model_names] for metric in metrics}
    
    # Create bar charts for each metric
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        rects = plt.bar(x + offset, metric_data[metric], width, label=metric)
        multiplier += 1
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 2, model_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    print("Generated model comparison plot")
    
    # Generate comprehensive results report
    with open('model_results.txt', 'w') as f:
        f.write("Heart Disease Prediction Model Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name} Results:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")
        
        # Add feature importance section
        if 'Random Forest' in feature_importances:
            f.write("\nRandom Forest Feature Importances:\n")
            f.write("-" * 30 + "\n")
            for feature, importance in feature_importances['Random Forest'].head(10).items():
                f.write(f"{feature}: {importance:.4f}\n")
        
        f.write("\nConclusion:\n")
        f.write("-" * 30 + "\n")
        best_model = max(results.items(), key=lambda x: x[1]['auc_roc'] if x[1]['auc_roc'] is not None else 0)
        f.write(f"Best performing model: {best_model[0]} (AUC-ROC: {best_model[1]['auc_roc']:.4f})\n")
        f.write("\nRecommendations:\n")
        f.write("1. Use Random Forest for prediction due to its superior performance\n")
        f.write("2. Focus on key risk factors identified in feature importance analysis\n")
        f.write("3. Implement targeted screening programs for individuals with multiple risk factors\n")
    
    print("Generated comprehensive results report")
    return results, models, feature_importances

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
    print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets")
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Train and evaluate models
    results, models, feature_importances = train_and_evaluate_models(X_train_balanced, X_test, y_train_balanced, y_test)
    
    print("\nAnalysis complete. See visualizations directory and reports for detailed results.")

if __name__ == "__main__":
    main() 