"""
IBM HR Analytics Employee Attrition Analysis
Comprehensive analysis including EDA, Feature Engineering, and ML Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class HRAttritionAnalysis:
    def __init__(self, data_path):
        """Initialize the analysis with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        
    def load_data(self):
        """Load the HR dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        return self.df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Attrition distribution
        print("\nAttrition Distribution:")
        print(self.df['Attrition'].value_counts())
        print(f"\nAttrition Rate: {(self.df['Attrition'].value_counts()['Yes'] / len(self.df)) * 100:.2f}%")
        
        return self.df
    
    def visualize_data(self):
        """Create comprehensive visualizations"""
        print("\nGenerating visualizations...")
        
        # Attrition by Department
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='Department', hue='Attrition')
        plt.title('Attrition by Department')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('attrition_by_department.png')
        plt.close()
        
        # Age distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='Age', hue='Attrition', kde=True, bins=30)
        plt.title('Age Distribution by Attrition')
        plt.tight_layout()
        plt.savefig('age_distribution.png')
        plt.close()
        
        # Monthly Income by Attrition
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='Attrition', y='MonthlyIncome')
        plt.title('Monthly Income by Attrition')
        plt.tight_layout()
        plt.savefig('income_by_attrition.png')
        plt.close()
        
        # Job Satisfaction
        plt.figure(figsize=(10, 6))
        pd.crosstab(self.df['JobSatisfaction'], self.df['Attrition']).plot(kind='bar')
        plt.title('Job Satisfaction vs Attrition')
        plt.xlabel('Job Satisfaction Level')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('satisfaction_vs_attrition.png')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(14, 10))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation = self.df[numeric_cols].corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        print("Visualizations saved!")
    
    def feature_engineering(self):
        """Prepare features for modeling"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Create a copy
        df_model = self.df.copy()
        
        # Encode target variable
        df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        categorical_cols = df_model.select_dtypes(include=['object']).columns
        
        le = LabelEncoder()
        for col in categorical_cols:
            df_model[col] = le.fit_transform(df_model[col])
        
        # Remove unnecessary columns
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        cols_to_drop = [col for col in cols_to_drop if col in df_model.columns]
        df_model = df_model.drop(columns=cols_to_drop)
        
        # Separate features and target
        X = df_model.drop('Attrition', axis=1)
        y = df_model['Attrition']
        
        print(f"\nFeature shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        print("2. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Gradient Boosting
        print("3. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        
        # XGBoost
        print("4. Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        
        print("\nAll models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
            print(f"\nConfusion Matrix:\n{confusion_matrix(self.y_test, y_pred)}")
            
            results[name] = {
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.close()
        
        print("\nROC curves saved!")
        
        return results
    
    def generate_insights(self):
        """Generate business insights"""
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS")
        print("="*50)
        
        # Top features from Random Forest
        rf_model = self.models.get('Random Forest')
        if rf_model:
            feature_importance = pd.DataFrame({
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        # Key statistics
        print("\nKey Statistics:")
        print(f"Average Age: {self.df['Age'].mean():.2f} years")
        print(f"Average Monthly Income: ${self.df['MonthlyIncome'].mean():.2f}")
        print(f"Average Years at Company: {self.df['YearsAtCompany'].mean():.2f} years")
        
        # Recommendations
        print("\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        print("""
        1. Focus on employee work-life balance and job satisfaction
        2. Review compensation packages for competitive alignment
        3. Implement retention programs for high-risk employees
        4. Conduct regular engagement surveys
        5. Provide career development opportunities
        6. Monitor overtime and work-life balance indicators
        """)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "#"*50)
        print("# IBM HR ANALYTICS - COMPLETE ANALYSIS")
        print("#"*50)
        
        self.load_data()
        self.exploratory_analysis()
        self.visualize_data()
        self.feature_engineering()
        self.train_models()
        self.evaluate_models()
        self.generate_insights()
        
        print("\n" + "#"*50)
        print("# ANALYSIS COMPLETE!")
        print("#"*50)

if __name__ == "__main__":
    # Run analysis
    analysis = HRAttritionAnalysis('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    analysis.run_complete_analysis()
