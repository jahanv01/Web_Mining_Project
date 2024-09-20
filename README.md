# Web_Mining_Project
# **Sentiment Analysis on Amazon Product Reviews**

### **Team Members:**
- **Jahanvi Panchal** - 1939439
- **Avani Gandhi** - 1937923
- **Manasi Patil** - 2034414
- **Saman Khursheed** - 1911156

**University of Mannheim**

---

## **Project Overview**

This project performs **sentiment analysis** on Amazon product reviews, focusing on the **Electronics** category. By applying **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, we classify product reviews as **positive**, **negative**, or **neutral**. Various methods, including **ML classifiers**, **lexicon-based approaches**, and **transformer models (BERT, GPT-3.5)**, were used to analyze the sentiment.

### **Key Features**
- **Data Preprocessing:** Text cleaning and preparation.
- **Machine Learning Classifiers:** Naive Bayes, SVM, Random Forest.
- **Lexicon-Based Sentiment Analysis:** VADER for polarity classification.
- **Advanced Models:** Transformer models such as BERT and GPT-3.5.
- **Evaluation Metrics:** F1-score, Confusion Matrix, and Classification Report.

---

## **Dataset**

We used the **Amazon US Customer Reviews Dataset** available on [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). Our focus was on **electronics product reviews**, with a subset of **10,000 reviews** selected for analysis.

### **Attributes:**
- **Star Rating:** Rating on a scale of 1-5, mapped to sentiment (positive, neutral, or negative).
- **Review Body:** Full review text.
- **Product Category:** The category of the product.
- **Helpful Votes:** Number of helpful votes for the review.

---

## **Methodology**

### 1. **Data Preprocessing**
- **Text Normalization:** Convert text to lowercase.
- **Tokenization:** Split the text into individual words.
- **Stopword Removal:** Remove common words (e.g., “the,” “and”).
- **Stemming:** Reduce words to their root form.

### 2. **Vectorization**
- **Count Vectorization:** Converts text into a bag-of-words model.
- **TF-IDF:** Weighs words based on importance.
- **FastText:** Embedding technique to capture word meanings and relationships.

### 3. **Modeling**
- **Classical ML Models:** Naive Bayes, SVM, Random Forest with hyperparameter tuning.
- **Lexicon-Based Models:** VADER for rule-based sentiment classification.
- **Transformer Models:** BERT and GPT-3.5 for deep language modeling.

### 4. **Model Evaluation**
- **F1-score:** Used to measure the balance between precision and recall.
- **Confusion Matrix & Classification Report**: Used to assess the accuracy of the model predictions, particularly for imbalanced classes.

---

## **Results**

### **Best Performing Models:**
- **GPT-3.5**: Weighted **F1-score: 0.87**
- **BERT**: Weighted **F1-score: 0.86**
- **Random Forest** (with oversampling): Weighted **F1-score: 0.81**

Both GPT-3.5 and BERT outperformed classical machine learning methods in predicting the sentiment of reviews.

---

## **Conclusion**

This project demonstrated that transformer models, such as **GPT-3.5** and **BERT**, provide superior performance in sentiment analysis tasks compared to traditional ML models like SVM and Random Forest. The next steps could involve performing **aspect-based sentiment analysis**, which focuses on specific product features (e.g., **Design, Performance, Durability**).

---

## **References**
1. **Joulin, A., et al. (2016).** FastText: Compressing text classification models. Available at [arXiv](https://arxiv.org/abs/1612.03651).
2. **Mahgoub, A., et al. (2022).** Sentiment analysis using BERT and TextBlob. IEEE.
