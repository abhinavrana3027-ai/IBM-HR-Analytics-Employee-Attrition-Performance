-- IBM HR Analytics SQL Queries
-- Business Intelligence queries for employee attrition analysis

-- ========================================
-- 1. OVERALL ATTRITION STATISTICS
-- ========================================

-- Total employees and attrition count
SELECT 
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data;

-- ========================================
-- 2. ATTRITION BY DEPARTMENT
-- ========================================

SELECT 
    Department,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY Department
ORDER BY attrition_rate DESC;

-- ========================================
-- 3. ATTRITION BY JOB ROLE
-- ========================================

SELECT 
    JobRole,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_monthly_income
FROM hr_employee_data
GROUP BY JobRole
ORDER BY attrition_rate DESC;

-- ========================================
-- 4. INCOME ANALYSIS BY ATTRITION
-- ========================================

SELECT 
    Attrition,
    COUNT(*) as employee_count,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(MIN(MonthlyIncome), 2) as min_income,
    ROUND(MAX(MonthlyIncome), 2) as max_income,
    ROUND(AVG(PercentSalaryHike), 2) as avg_salary_hike
FROM hr_employee_data
GROUP BY Attrition;

-- ========================================
-- 5. DISTANCE FROM HOME ANALYSIS
-- ========================================

SELECT 
    CASE 
        WHEN DistanceFromHome <= 5 THEN 'Very Close (0-5)'
        WHEN DistanceFromHome <= 10 THEN 'Close (6-10)'
        WHEN DistanceFromHome <= 20 THEN 'Moderate (11-20)'
        ELSE 'Far (20+)'
    END as distance_category,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY distance_category
ORDER BY attrition_rate DESC;

-- ========================================
-- 6. AGE GROUP ANALYSIS
-- ========================================

SELECT 
    CASE 
        WHEN Age < 25 THEN 'Under 25'
        WHEN Age BETWEEN 25 AND 34 THEN '25-34'
        WHEN Age BETWEEN 35 AND 44 THEN '35-44'
        WHEN Age BETWEEN 45 AND 54 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY age_group
ORDER BY age_group;

-- ========================================
-- 7. YEARS AT COMPANY ANALYSIS
-- ========================================

SELECT 
    CASE 
        WHEN YearsAtCompany <= 2 THEN '0-2 years'
        WHEN YearsAtCompany <= 5 THEN '3-5 years'
        WHEN YearsAtCompany <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END as tenure_group,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY tenure_group
ORDER BY attrition_rate DESC;

-- ========================================
-- 8. JOB SATISFACTION AND ATTRITION
-- ========================================

SELECT 
    JobSatisfaction,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY JobSatisfaction
ORDER BY JobSatisfaction;

-- ========================================
-- 9. WORK-LIFE BALANCE ANALYSIS
-- ========================================

SELECT 
    WorkLifeBalance,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income
FROM hr_employee_data
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance;

-- ========================================
-- 10. OVERTIME AND ATTRITION
-- ========================================

SELECT 
    OverTime,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY OverTime;

-- ========================================
-- 11. EDUCATION FIELD ANALYSIS
-- ========================================

SELECT 
    EducationField,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income
FROM hr_employee_data
GROUP BY EducationField
ORDER BY attrition_rate DESC;

-- ========================================
-- 12. HIGH-RISK EMPLOYEE IDENTIFICATION
-- ========================================

SELECT 
    EmployeeNumber,
    Age,
    Department,
    JobRole,
    MonthlyIncome,
    YearsAtCompany,
    JobSatisfaction,
    WorkLifeBalance,
    OverTime,
    DistanceFromHome
FROM hr_employee_data
WHERE Attrition = 'No'
    AND (
        JobSatisfaction <= 2 OR
        WorkLifeBalance <= 2 OR
        OverTime = 'Yes' OR
        YearsAtCompany <= 2
    )
ORDER BY JobSatisfaction, YearsAtCompany;

-- ========================================
-- 13. MARITAL STATUS AND ATTRITION
-- ========================================

SELECT 
    MaritalStatus,
    Gender,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY MaritalStatus, Gender
ORDER BY attrition_rate DESC;

-- ========================================
-- 14. TRAINING TIMES AND ATTRITION
-- ========================================

SELECT 
    CASE 
        WHEN TrainingTimesLastYear = 0 THEN 'No Training'
        WHEN TrainingTimesLastYear <= 2 THEN '1-2 times'
        WHEN TrainingTimesLastYear <= 4 THEN '3-4 times'
        ELSE '5+ times'
    END as training_frequency,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate
FROM hr_employee_data
GROUP BY training_frequency
ORDER BY attrition_rate DESC;

-- ========================================
-- 15. PERFORMANCE RATING ANALYSIS
-- ========================================

SELECT 
    PerformanceRating,
    COUNT(*) as employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(PercentSalaryHike), 2) as avg_salary_hike
FROM hr_employee_data
GROUP BY PerformanceRating
ORDER BY PerformanceRating DESC;
