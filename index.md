# CYT180 — Lab 9: Email Spam Filtering 
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** 


----

## Introduction
In this lab, you will run, understand, and document an end‑to‑end machine learning pipeline that classifies
email messages as **spam** or **ham (harmless)**. The work is based on an open‑source project (GPL‑licensed)
and a provided Jupyter Notebook. Your focus is on **comprehension and analysis**: running the code, interpreting outputs, 
and explaining each step in your own words.

In this lab you will learn about:
- The supervised learning workflow for **binary classification**
- How text is transformed using **NLP preprocessing** (cleaning, tokenization, stopwords, stemming)
- How **Bag‑of‑Words** (CountVectorizer) converts text into numeric features
- Training and comparing three models:
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
  - **Multinomial Naïve Bayes**
- Evaluating models using **accuracy, precision, recall, F1‑score, and confusion matrix**
- Saving the **best model** and using it to **predict** on new messages

----

## 📂 Source Files and Materials

You will use the following resources for this lab:

### **1. GitHub Project (Open Source, GPL License)**
This lab is based on the following open‑source project:

🔗 **Email Spam Detection Repository**  
https://github.com/kanagalingamsm/Email-Spam-Detection

This project provides the main notebook and pipeline used in Lab 9.

---

### **2. Jupyter Notebook**
A version of the notebook (`Email-Spam-Detection.ipynb`) is posted on Blackboard.  
Students must download it and upload it into **Google Colab**.

---

### **3. Dataset — `spam.csv`**
This dataset contains labeled email messages (spam or ham).  
You will upload this file into Colab along with the notebook.

---

### **4. Additional Lab Instructions**
A copy of the Lab 9 instructions (this document) will be provided on:

- **Blackboard**
- **GitHub (this repository)**

Students must follow both sources while completing the lab.

---

### **Summary of Required Items**
- Notebook: `Email-Spam-Detection.ipynb`  
- Dataset: `spam.csv`  
- This Lab 9 instruction document  
- GitHub project reference (read for understanding)

----
## 🎯 Lab Objectives

By the end of this lab, you will be able to:

1. **Explain the end-to-end ML workflow** for a **binary classification** problem (spam vs. ham).
2. **Run and interpret** an NLP preprocessing pipeline:
   - Cleaning text (remove non-letters, lowercase)
   - Tokenization
   - Stopword removal
   - Stemming (PorterStemmer)
   - Building a **corpus**
3. **Convert text to numeric features** using **Bag-of-Words** with `CountVectorizer`.
4. **Train and compare** three classifiers:
   - Decision Tree Classifier
   - Random Forest Classifier
   - Multinomial Naïve Bayes
5. **Evaluate models** using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - **Confusion Matrix** (TP, TN, FP, FN)
6. **Select and save** the best-performing model to a file and **load it back** to make predictions.
7. **Classify unseen messages** by applying the same preprocessing and using the saved model.
8. **Document your understanding** with clear, concise explanations and annotated screenshots in a PDF report.
