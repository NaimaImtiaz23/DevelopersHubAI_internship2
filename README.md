# DevelopersHubAI_internship2

**Author:** Naima Imtiaz  
**Internship Role:** AI/ML Engineering Intern  
**Organization:** DevelopersHub Corporation  

---

## Overview

This repository contains the implementation of selected advanced AI/ML tasks completed as part of the DevelopersHub AI/ML Engineering Internship. The objective of these tasks is to develop practical expertise in building real-world machine learning systems using modern tools and frameworks.

The work focuses on:
- Building production-ready ML pipelines  
- Applying multimodal learning techniques  
- Developing context-aware conversational AI systems  

The following tasks have been successfully completed:

- Task 2: End-to-End ML Pipeline for Customer Churn Prediction  
- Task 3: Multimodal Housing Price Prediction (Image + Tabular Data)  
- Task 4: Context-Aware Chatbot using LangChain (RAG-based)  

---

## Task 2: End-to-End ML Pipeline for Customer Churn Prediction

### Objective
To design and implement a scalable and reusable machine learning pipeline for predicting customer churn using structured telecom data.

### Dataset
Telco Customer Churn Dataset

### Approach

#### Data Preprocessing
- Cleaned dataset and handled missing values  
- Encoded categorical variables  
- Scaled numerical features  

#### Pipeline Development
- Built an integrated pipeline using Scikit-learn Pipeline API  
- Combined preprocessing and model training steps  

#### Model Training
- Logistic Regression  
- Random Forest Classifier  

#### Hyperparameter Tuning
- Applied GridSearchCV for model optimization  
- Performed cross-validation for reliable performance  

#### Model Export
- Saved the final trained pipeline using joblib for reuse  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  

### Key Results
- Random Forest achieved better performance due to handling complex patterns  
- Pipeline ensures reproducibility and production readiness  

### Skills Developed
- Machine learning pipelines  
- Feature preprocessing and engineering  
- Hyperparameter tuning  
- Model serialization  

---

## Task 3: Multimodal ML – Housing Price Prediction

### Objective
To predict housing prices by combining structured data with image-based features using a multimodal learning approach.

### Dataset
- Housing dataset (tabular features)  
- House image dataset  

### Approach

#### Image Feature Extraction
- Used Convolutional Neural Networks (CNNs) to extract visual features  
- Converted images into feature vectors  

#### Tabular Data Processing
- Cleaned and normalized structured data  
- Selected relevant predictive features  

#### Feature Fusion
- Combined image features with tabular data  
- Created a unified dataset for training  

#### Model Training
- Trained regression models on combined features  

### Evaluation Metrics
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

### Key Results
- Multimodal approach improved prediction performance  
- Image data enhanced understanding of property conditions  

### Skills Developed
- Multimodal machine learning  
- CNN-based feature extraction  
- Feature fusion techniques  
- Regression modeling  

---

## Task 4: Context-Aware Chatbot Using LangChain (RAG)

### Objective
To build a chatbot capable of maintaining conversational context and retrieving relevant information using Retrieval-Augmented Generation (RAG).

### Dataset
Custom knowledge base consisting of AI/ML-related documents.

### Approach

#### Document Processing
- Split text into smaller chunks for efficient processing  
- Generated embeddings using pre-trained models  

#### Vector Database
- Stored embeddings using FAISS  
- Enabled similarity-based document retrieval  

#### Conversational Memory
- Implemented memory to retain chat history  
- Enabled context-aware responses  

#### Retrieval-Augmented Generation
- Retrieved relevant documents before generating responses  
- Improved accuracy using external knowledge  

#### Model Integration
- Used Hugging Face transformer model  
- Integrated with LangChain for full pipeline  

#### Deployment
- Implemented chatbot interface (CLI-based, extendable to Streamlit)  

### Key Features
- Context-aware conversations  
- Document-based answer retrieval  
- Improved response accuracy using RAG  

### Skills Developed
- Conversational AI  
- LangChain framework  
- Vector databases and embeddings  
- Retrieval-Augmented Generation (RAG)  
- LLM integration  

---

---

## Conclusion

This repository demonstrates the implementation of advanced machine learning solutions across multiple domains, including structured data modeling, multimodal learning, and conversational AI.

The completed tasks highlight practical skills required for real-world AI/ML systems, including scalability, performance optimization, and integration of modern frameworks.

---
