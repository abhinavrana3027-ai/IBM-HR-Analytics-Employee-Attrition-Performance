# IBM HR Analytics: Employee Attrition & Performance Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview

Employee attrition is a critical challenge for organizations, with the average cost of replacing an employee estimated at **50-200% of their annual salary**. This project provides a comprehensive analysis of the IBM HR Analytics dataset to uncover the key factors driving employee turnover and build predictive models to identify at-risk employees.

### Business Problem

Organizations face significant challenges when employees leave:
- **Direct costs**: Recruitment, hiring, and training expenses
- **Indirect costs**: Lost productivity, knowledge drain, and team morale impact
- **Strategic impact**: Disruption to ongoing projects and customer relationships

This analysis aims to help HR teams proactively identify and retain valuable employees by understanding the factors that contribute to attrition.

## Key Findings

### Attrition Overview
- **Overall Attrition Rate**: 16.1% (237 out of 1,470 employees)
- **High-Risk Departments**: Sales (20.6%), Human Resources (19.0%)
- **Job Roles with Highest Attrition**: Sales Representative (39.8%), Laboratory Technician (23.9%)

### Top Factors Influencing Attrition

| Factor | Impact Level | Key Insight |
|--------|-------------|-------------|
| Overtime | High | Employees working overtime are **2.8x more likely** to leave |
| Monthly Income | High | Lower-income employees show significantly higher attrition |
| Job Satisfaction | High | Low satisfaction correlates with **2.5x higher** attrition |
| Work-Life Balance | Medium | Poor work-life balance increases attrition risk by **67%** |
| Years at Company | Medium | Employees with <2 years tenure are most at risk |
| Business Travel | Medium | Frequent travelers show **3x higher** attrition rates |

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|----------|
| Random Forest | 87.4% | 0.72 | 0.65 | 0.68 | 0.84 |
| XGBoost | 88.1% | 0.74 | 0.68 | 0.71 | 0.86 |
| Logistic Regression | 84.2% | 0.65 | 0.58 | 0.61 | 0.79 |
| Gradient Boosting | 87.8% | 0.73 | 0.66 | 0.69 | 0.85 |

## Dataset Description

The dataset contains **1,470 employee records** with **35 features** covering:

### Employee Demographics
- `Age`: Employee age (18-60 years)
- `Gender`: Male/Female
- `MaritalStatus`: Single, Married, Divorced
- `Education`: 1-5 scale (Below College to Doctor)
- `EducationField`: Life Sciences, Medical, Marketing, etc.

### Job Information
- `JobRole`: 9 different roles including Sales Executive, Research Scientist, etc.
- `Department`: Sales, R&D, Human Resources
- `JobLevel`: 1-5 hierarchy level
- `JobInvolvement`: 1-4 scale
- `JobSatisfaction`: 1-4 scale

### Compensation & Benefits
- `MonthlyIncome`: Monthly salary
- `DailyRate`, `HourlyRate`: Pay rates
- `PercentSalaryHike`: Recent salary increase percentage
- `StockOptionLevel`: 0-3 stock option level

### Work Experience
- `TotalWorkingYears`: Total years of work experience
- `YearsAtCompany`: Tenure at current company
- `YearsInCurrentRole`: Time in current position
- `YearsSinceLastPromotion`: Time since last promotion
- `YearsWithCurrManager`: Time with current manager
- `NumCompaniesWorked`: Previous employers count

### Work Environment
- `BusinessTravel`: Travel frequency
- `DistanceFromHome`: Commute distance
- `OverTime`: Yes/No
- `WorkLifeBalance`: 1-4 scale
- `EnvironmentSatisfaction`: 1-4 scale
- `RelationshipSatisfaction`: 1-4 scale

### Performance
- `PerformanceRating`: 1-4 scale (3-4 observed)
- `TrainingTimesLastYear`: Training sessions attended

### Target Variable
- `Attrition`: Yes/No (Binary classification target)

## Project Structure

```
IBM-HR-Analytics-Employee-Attrition-Performance/
|
|-- data/
|   |-- WA_Fn-UseC_-HR-Employee-Attrition.csv    # Raw dataset
|
|-- notebooks/
|   |-- IBM_HR_Analytics_Complete_Analysis.ipynb  # Main analysis notebook
|
|-- src/
|   |-- data_preprocessing.py                     # Data cleaning functions
|   |-- feature_engineering.py                    # Feature creation
|   |-- model_training.py                         # ML model functions
|   |-- visualization.py                          # Plotting utilities
|
|-- sql/
|   |-- hr_analytics_queries.sql                  # SQL analysis queries
|
|-- reports/
|   |-- executive_summary.md                      # Business summary
|
|-- requirements.txt                              # Project dependencies
|-- README.md                                     # Project documentation
|-- LICENSE                                       # MIT License
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/abhinavrana3027-ai/IBM-HR-Analytics-Employee-Attrition-Performance.git
cd IBM-HR-Analytics-Employee-Attrition-Performance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/IBM_HR_Analytics_Complete_Analysis.ipynb
```

## Analysis Methodology

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis of all features
- Bivariate analysis with attrition
- Correlation analysis and multicollinearity check
- Distribution analysis and outlier detection

### 2. Data Preprocessing
- Handling missing values (none found in this dataset)
- Encoding categorical variables (Label & One-Hot Encoding)
- Feature scaling (StandardScaler for numerical features)
- Handling class imbalance (SMOTE oversampling)

### 3. Feature Engineering
- Created composite features:
  - `Income_Per_Year_Experience`: MonthlyIncome / TotalWorkingYears
  - `Promotion_Stagnation`: YearsAtCompany - YearsSinceLastPromotion
  - `Satisfaction_Score`: Average of all satisfaction metrics
  - `Career_Growth_Rate`: JobLevel / YearsAtCompany

### 4. Model Development
- Train-test split (80-20) with stratification
- Cross-validation (5-fold) for robust evaluation
- Hyperparameter tuning using GridSearchCV
- Model comparison and selection

### 5. Model Interpretation
- Feature importance analysis
- SHAP values for model explainability
- Partial Dependence Plots
- Business recommendations based on insights

## Key Visualizations

### Attrition by Department
```
Sales           |#################### 20.6%
Human Resources |################### 19.0%
R&D             |############# 13.8%
```

### Attrition by Overtime
```
Yes (Overtime)  |############################## 30.5%
No (No OT)      |########## 10.4%
```

### Monthly Income Distribution
- Employees who left had **31% lower** median income than those who stayed
- Largest gap observed in Sales and R&D departments

## Business Recommendations

### Immediate Actions
1. **Address Overtime**: Implement workload balancing and hire additional staff in high-overtime departments
2. **Review Compensation**: Conduct salary benchmarking, especially for Sales Representatives and Lab Technicians
3. **Improve Work-Life Balance**: Introduce flexible working arrangements and wellness programs

### Medium-term Strategies
4. **Career Development**: Create clear promotion pathways and skill development programs
5. **Manager Training**: Invest in leadership training as manager relationships impact retention
6. **Travel Policy**: Review business travel requirements and offer alternatives

### Long-term Initiatives
7. **Predictive Monitoring**: Implement the attrition prediction model in HRIS
8. **Stay Interviews**: Conduct regular check-ins with high-risk employees
9. **Culture Enhancement**: Build a culture of recognition and employee engagement

## Future Enhancements

- [ ] Build interactive dashboard using Streamlit/Dash
- [ ] Implement real-time prediction API
- [ ] Add survival analysis for time-to-attrition prediction
- [ ] Integrate with HRIS systems for automated risk scoring
- [ ] A/B testing framework for retention interventions

## Technologies Used

- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Model Interpretation**: SHAP, Lime
- **Statistical Analysis**: SciPy, Statsmodels
- **Database**: SQL (for query demonstrations)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IBM Data Science Team for creating and sharing this dataset
- Kaggle community for inspiration and insights
- Open source community for amazing tools and libraries

## Contact

**Abhinav Rana**
- GitHub: [@abhinavrana3027-ai](https://github.com/abhinavrana3027-ai)
- LinkedIn: [Connect with me](https://linkedin.com/in/abhinavrana)

---

*If you find this project useful, please consider giving it a star!*
