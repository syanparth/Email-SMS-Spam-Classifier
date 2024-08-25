# Spam Classifier Project

This repository contains the implementation of a spam classifier using machine learning techniques. The project involves data cleaning, exploration, feature extraction, model training, and evaluation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Cleaning](#data-cleaning)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)

## Project Overvie

This project aims to develop a spam classifier that can differentiate between spam and ham messages using machine learning techniques.

## Data Cleaning

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("spam.csv")

# Clean the data
df.dropna(inplace=True)
df = df[["label", "message"]]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
