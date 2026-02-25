# CYT180 — Lab 9: Email Spam Filtering 
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** PDF Report

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

For this lab you will use an **Open Source, GPL License GitHub Project** as mentioned below.

**Email Spam Detection Repository** https://github.com/kanagalingamsm/Email-Spam-Detection

This project provides the main notebook and pipeline used in Lab 9.
You will use the following files from this project:
- **Jupyter Notebook:** Download the `Email-Spam-Detection.ipynb` and upload it into **Google Colab**.
- **Dataset:** Download the file`spam.csv`. This dataset contains labeled email messages (spam or ham). You will upload this file into Colab along with the notebook.
- **README.md:** This file contains the project description, you must reffer to this to understand the project.

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

## Environment Setup

- This lab is designed to run in **Google Colab**. You can run this in local Anaconda jupyter notebook as well, but it will take some time for the processing since the data file is big.
- Start from the downloaded notebook `Email-Spam-Detection.ipynb`. Make a personal copy of it and rename it to: `CYT180_Lab9_YourName.ipynb`. Work in your renamed copy for the entire lab.
 
----

## Getting Started: Step‑by‑Step Lab Workflow

You will follow the provided Jupyter Notebook **exactly as written**.  
For each block of code, you must:

1. **Learn** — Read the theory and understand the purpose  
2. **Run** — Execute the code in Google Colab  
3. **Comment** — Write your own explanation in the markdown cell

Your final submission PDF must contain **screenshots + explanations** for each of the major steps below.

----

### Step 1 — Load and Inspect the Dataset (EDA)

**Learn**
- Understand the structure of `spam.csv`.
- What columns exist (`label`, `message`)?
- How many spam vs. ham messages?
- Why do we inspect the dataset before preprocessing?

**Run**
- Load the dataset
- Display the first few rows
- Show dataset shape
- Count spam vs. ham messages
- Check for missing values

**Comment:** Explain in your own words:
- What the dataset looks like  
- What “spam” and “ham” represent  
- Any initial observations (class imbalance, duplicates, etc.)

-----

### Step 2 — Clean and Preprocess the Text (NLP Pipeline)

**Learn:** This section contains core NLP concepts:
- Removing non-alphabetic characters
- Converting to lowercase
- Splitting text into tokens
- Removing stopwords
- Stemming words
- Constructing a corpus

**Run:** Execute the preprocessing code, such as:
- Looping through each message  
- Using `re.sub('[^a-zA-Z]', ' ', ...)`  
- Lowercasing  
- Tokenizing  
- Removing stopwords (`stopwords.words('english')`)  
- Stemming (`PorterStemmer`)  
- Building the final cleaned corpus  

**Comment:** Explain each preprocessing step:
- Why remove punctuation?
- Why convert to lowercase?
- What are stopwords?
- What does stemming do?
- What is a “corpus”?

----

### Step 3 — Feature Extraction (Bag‑of‑Words)

**Learn:**
- Understand why text must be converted to numbers  
- What is **CountVectorizer**  
- What `Bag‑of‑Words` means  
- What `max_features=4000` does  

**Run:**
- Execute the corresponding python cell containing
```python
  cv = CountVectorizer(max_features=4000)
  X = cv.fit_transform(corpus).toarray()
```

**Comment:**
- What the feature matrix X represents
- Why we limit vocabulary size
- The shape of the matrix (rows = messages, columns = words)
- Your understanding of `CountVectorizer` and `Bag‑of‑Words`

----

### Step 4 — Prepare Labels (Y)

**Learn:**
- Understand why  labels must be numeric  
- How pd.get_dummies() works
- Why we use one column (binary classification)

**Run:**
- Execute the corresponding python cell
```python
  Y = pd.get_dummies(spam['label'])
  Y = Y.iloc[:, 1].values
```

**Comment:**
- What pd.get_dummies() creates
- Why we select the second column
- Meaning of:
  - 1 = spam
  - 0 = ham
 
----

### Step 5 — Train/Test Split

**Learn:**
- Understand why we split into training and testing sets
- What overfitting means
- Why 70/30 or 80/20 splits are common

**Run:**
- Execute the corresponding python cell
```python
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
```

**Comment:**
- What train_test_split does
- Why we need separate training and test sets

----

### Step 6 — Train Three ML Models
You will run and compare:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Multinomial Naïve Bayes

**Learn:** For each model, understand:
- What type of classifier it is
- Why it might perform well or poorly on text data

**Run:**
- Execute the corresponding python cell containing code like this.
```python
  dt = DecisionTreeClassifier().fit(X_train, Y_train)
  rf = RandomForestClassifier().fit(X_train, Y_train)
  nb = MultinomialNB().fit(X_train, Y_train)
```

**Comment:** For each model include:
- Screenshot of accuracy score
- Short explanation (2–3 sentences):
  - What the model is
  - Its accuracy compared to others

----

### Step 7 — Evaluate the Models

**Learn:** Understand the metrics included
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (TP, TN, FP, FN)

**Run:**
- Execute the corresponding python cell containing code like this.
```python
pred = model.predict(X_test)
print(accuracy_score(Y_test, pred))
print(precision_score(Y_test, pred))
print(recall_score(Y_test, pred))
print(f1_score(Y_test, pred))
print(confusion_matrix(Y_test, pred))
```

**Comment:** Exaplain
- What each metric tells you
- What the confusion matrix means (4 quadrants)
- Why precision/recall may matter more than accuracy in spam filtering

----
### Step 8 — Select the Best Model, Save and Load the Best Model

- Comment Why one model outperforms others.   
- Which model should be selected so that we can save it and use it for new data. Justify the chosen best model.
- Save the selected model using `pickle`:
  
``` python
pickle.dump(best_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
```
----

### Step 9 — Predict on a New Email Message

- Use whatever prediction cell is included in the notebook and try your own message.

----

## Submission Guidelines

Your submission must include a **PDF report** containing screenshots and explanations.  
Follow all instructions carefully — submissions that do not meet these requirements may not be accepted.
Your PDF must include the following sections:

1. **Title Page**
- Course ID: **CYT180**
- Lab Number: **Lab 9 – Email Spam Filtering**
- Your full name
- Date

2. **Screenshots from Your Notebook**
- Include screenshots for every major step of the lab done above. 
- Also include the answers of questions from each step's "comment" section. These answers can be added as screenshots from your notebooks markdown cells or added as text in the final pdf.
- Screenshots must be clear and readable. Partial or missing screenshots result in reduced marks.

3. **Final Conclusion**
Write 1–2 short paragraphs summarizing:
- What you learned  
- How ML is applied to spam filtering  
- Strengths and limitations of the approach  
