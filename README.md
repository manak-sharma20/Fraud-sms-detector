
# Online Scam Message Detection

## Overview
This project detects online scam messages using Natural Language Processing and Machine Learning.

## Dataset
SMS Spam Collection Dataset (Kaggle)

## Approach
- Text preprocessing (lowercasing, URL removal, punctuation removal)
- TF-IDF vectorization (unigrams and bigrams)
- Multinomial Naive Bayes classifier
- Model evaluation using accuracy, precision, recall, and confusion matrix

## Results
The model achieved high accuracy and precision, effectively identifying scam messages.

## Example
Input:
"You have won ₹50,000. Click link now"

Output:
Scam
