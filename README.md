I worked on a machine-learning project called Stroke Prediction, where the objective was to predict whether a patient had a risk of stroke based on demographic and physiological factors.

I used a healthcare stroke dataset containing 5,110 patient records and 12 variables, including age, gender, hypertension, heart disease, average glucose level, BMI, smoking status and the stroke outcome.

I started by loading and understanding the data using Pandas, and performed data-quality checks using info(), describe() and missing-value analysis. I removed the patient ID because it was only an identifier and did not provide meaningful predictive information.

The main missing-data issue was in the BMI column, where there were 201 missing values. I examined the BMI distribution and used mean imputation to handle those missing values.

Next, I identified the categorical variables such as gender, work type, residence type and smoking status, and converted them into numerical values using Label Encoding. I then separated the independent variables from the target variable, where stroke was the target, and applied Min-Max scaling to bring the features into a common 0-to-1 range.

After preprocessing, I performed Exploratory Data Analysis using Matplotlib and Seaborn. One important finding was that the dataset was highly imbalanced, with significantly more non-stroke cases than stroke cases.

For machine learning, I used four classification models: Support Vector Machine, Logistic Regression, K-Nearest Neighbors and Random Forest. I split the data into training and testing sets using stratification to preserve the class distribution.

I evaluated the models using accuracy, precision, recall, F1-score and confusion matrices, rather than relying only on accuracy because of the class imbalance.

Based on the results in my notebook, Random Forest showed the strongest reported performance, with 95% test accuracy, precision, recall and F1-score. I then explored GridSearchCV to fine-tune the Random Forest model.

The main learning from this project was not just building a prediction model, but understanding the complete data-science workflow — from data cleaning and preprocessing to EDA, model selection, evaluation and hyperparameter tuning
