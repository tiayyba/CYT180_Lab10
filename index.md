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

## Getting Started

This lab is designed to run in **Google Colab**. You can run this in local Anaconda jupyter notebook as well, but it will take some time for the processing since the data file is big.
Start from the downloaded notebook `Email-Spam-Detection.ipynb`. Make a personal copy of it and rename it to: `CYT180_Lab9_YourName.ipynb`. Work in your renamed copy for the entire lab.
 
----

## 🧪 Step‑by‑Step Lab Workflow

You will follow the provided Jupyter Notebook **exactly as written**.  
For each block of code, you must:

1. **Learn** — Read the theory and understand the purpose  
2. **Run** — Execute the code in Google Colab  
3. **Comment** — Write your own explanation in your final PDF report

Your final PDF must contain **screenshots + explanations** for each of the major steps below.

---

## 🔎 Step 1 — Load and Inspect the Dataset (EDA)

### Learn
Understand:
- What is the structure of `spam.csv`?
- What columns exist (`label`, `message`)?
- How many spam vs. ham messages?
- Why do we inspect the dataset before preprocessing?

### Run
Typical actions:
- Load the dataset
- Display the first few rows
- Show dataset shape
- Count spam vs. ham messages
- Check for missing values

### Comment
Explain in your own words:
- What the dataset looks like  
- What “spam” and “ham” represent  
- Any initial observations (class imbalance, duplicates, etc.)

---

## 🔧 Step 2 — Clean and Preprocess the Text (NLP Pipeline)

### Learn
This section contains core NLP concepts:
- Removing non-alphabetic characters
- Converting to lowercase
- Splitting text into tokens
- Removing stopwords
- Stemming words
- Constructing a corpus

### Run
Execute the preprocessing code, such as:

- Looping through each message  
- Using `re.sub('[^a-zA-Z]', ' ', ...)`  
- Lowercasing  
- Tokenizing  
- Removing stopwords (`stopwords.words('english')`)  
- Stemming (`PorterStemmer`)  
- Building the final cleaned corpus  

### Comment
Explain **each** preprocessing step:
- Why remove punctuation?
- Why convert to lowercase?
- What are stopwords?
- What does stemming do?
- What is a “corpus”?

Include screenshots of:
- A few processed messages
- The printed corpus (first 5 items)

---

## 🧮 Step 3 — Feature Extraction (Bag‑of‑Words)

### Learn
Understand:
- Why text must be converted to numbers  
- What is **CountVectorizer**  
- What “Bag‑of‑Words” means  
- What `max_features=4000` does  

### Run
Execute:
```python
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()
