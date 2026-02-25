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

## Source Files and Materials

You will use the following resources for this lab:

### GitHub Project (Open Source, GPL License)**
This lab is based on the following open‑source project:

**Email Spam Detection Repository**  
https://github.com/kanagalingamsm/Email-Spam-Detection

This project provides the main notebook and pipeline used in Lab 9.
You will use the following files from this project:
- **Jupyter Notebook:** Download the `Email-Spam-Detection.ipynb` and upload it into **Google Colab**.
- **Dataset:** Download the file`spam.csv`. This dataset contains labeled email messages (spam or ham). You will upload this file into Colab along with the notebook.
- **README.md** This file contains the project description, you must reffer to this to understand the project.

----

## Lab Objectives

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
   - Confusion Matrix (TP, TN, FP, FN)
6. **Select and save** the best-performing model to a file and **load it back** to make predictions.
7. **Classify unseen messages** by applying the same preprocessing and using the saved model.
8. **Document your understanding** with clear, concise explanations and annotated screenshots in a PDF report.

----
