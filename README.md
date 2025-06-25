# titanic
kaggel competition - titanic dataset
# 🚢 Titanic Survival Prediction (Kaggle ML Project)

This project predicts passenger survival using data from the Titanic disaster. It is built in Google Colab using various machine learning models and thorough data preprocessing and feature engineering.

[🔗 Kaggle Competition Link](https://www.kaggle.com/c/titanic)

---

## 📊 Project Overview

**Objective:** Predict which passengers survived the Titanic shipwreck using structured data such as age, gender, ticket class, and more.

**Dataset:**  
- Titanic dataset from Kaggle: `train.csv`, `test.csv`

---

## 🧰 Tools and Libraries

- **Platform:** Google Colab
- **Language:** Python 3
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn` (LogisticRegression, SVC, KNN, RandomForest, etc.)

---

## ⚙️ Workflow

### 📌 1. Data Preprocessing
- Filled missing values:
  - `Age`: imputed based on logic
  - `Embarked`: filled with mode
  - `Fare`: filled in test set
- Dropped irrelevant features: `Name`, `Cabin`, `Ticket`, `PassengerId`
- Converted categorical features:
  - `Sex` → binary
  - Binned continuous features like `Age` and `Fare`

### 🤖 2. Models Used
- **Logistic Regression**
- **Support Vector Machine (SVC)**
- **k-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Naive Bayes**
- **Decision Tree**
- And more (SGD, Perceptron, etc.)

### 🧪 3. Evaluation
- Compared performance using confidence scores
- Assessed models based on predictive accuracy

---

## ✅ Results

- All models were evaluated and compared for prediction confidence and accuracy
- Random forest perform the best  when evaluating with k-fold cross validation
- Random forest models' hyperparameters were furthur tuned to yield the best accuracy and generalization
- Final model performance showed solid predictive capability. Average accuracy of k-folds:0.8328, Test set competition accuracy: 0.78229

---

## 🧠 Key Learnings

- Practical experience in cleaning and transforming real-world data
- Applied and compared several ML algorithms on the same dataset
- Importance of preprocessing and encoding for classical ML models

---

## 🚀 How to Run
1. Open My_titanic.ipynb with Jupyter Notebook/Google Colab
2. Run all cells top to bottom

---

## 📬 Contact

**Shotitouch Tuangcharoentip**  
- [GitHub](https://github.com/shotitouch)  
- [LinkedIn](https://linkedin.com/in/shotitouch-tuangcharoentip-b3aa77159)

---

## 📜 License

This project is open-sourced under the MIT License.
