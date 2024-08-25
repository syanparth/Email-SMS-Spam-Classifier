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

## Project Overview

This project aims to develop a spam classifier that can differentiate between spam and ham messages using machine learning techniques.

## Data Cleaning

Data cleaning is a crucial step in preparing the dataset for analysis. It involves removing any inconsistencies, filling or removing missing values, and transforming the data into a format suitable for modeling. In this project, we focus on cleaning the data by:

1. **Loading the Dataset**: The dataset used for this spam classifier is a CSV file containing text messages labeled as either 'spam' or 'ham' (not spam).

2. **Removing Missing Values**: Any rows with missing data are removed to ensure that the dataset is complete and won't cause errors during processing.

3. **Selecting Relevant Columns**: The dataset might contain extra information that is not necessary for building the spam classifier. Here, we only keep the 'label' and 'message' columns.

4. **Label Encoding**: The 'label' column contains text labels ('ham' for non-spam and 'spam' for spam). These labels are converted into numerical values to make them suitable for machine learning algorithms. Specifically, 'ham' is encoded as 0, and 'spam' is encoded as 1.

```python
import numpy as np               # Importing NumPy for numerical operations
import pandas as pd              # Importing pandas for data manipulation
import matplotlib.pyplot as plt  # Importing Matplotlib for data visualization

# Step 1: Load the dataset
df = pd.read_csv("spam.csv")     # Reading the dataset from a CSV file into a pandas DataFrame

# Step 2: Clean the data
df.dropna(inplace=True)          # Removing any rows with missing values to ensure data integrity
df = df[["label", "message"]]    # Selecting only the columns 'label' and 'message' for further analysis
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encoding labels: 'ham' -> 0, 'spam' -> 1

