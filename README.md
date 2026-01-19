# Online Scam Message Detection

## Overview
This project focuses on detecting online scam messages using Natural Language Processing (NLP) and Machine Learning techniques. The model classifies messages as **scam** or **safe** based on textual patterns.

## Dataset
SMS Spam Collection Dataset (Kaggle), containing labeled SMS messages used for training and evaluation.

## Approach
- Text preprocessing including lowercasing, URL removal, number removal, and punctuation removal
- Feature extraction using TF-IDF vectorization (unigrams and bigrams)
- Classification using Multinomial Naive Bayes
- Model evaluation using accuracy, precision, recall, and confusion matrix
- Inference testing with real-world custom messages

## Results
The model achieved high accuracy and strong precision, demonstrating effective detection of scam messages while minimizing false safe predictions.

## Example
**Input:**  
"You have won ₹50,000. Click link now"

**Output:**  
Scam
