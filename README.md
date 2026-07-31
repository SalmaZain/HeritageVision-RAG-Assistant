# HeritageVision RAG Assistant 🏛️

An AI-powered heritage assistant that combines **Computer Vision**, **Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)** to identify and explain British architectural styles.

The system allows users to upload building images, classify architectural styles using a deep learning model, and receive contextual explanations through a conversational AI assistant powered by a heritage knowledge base.

---

# Project Overview

HeritageVision RAG Assistant was developed as a multimodal AI system for heritage knowledge support.

The system addresses two main tasks:

1. **Architectural Style Recognition**
   - Users upload an image of a building.
   - A Computer Vision model predicts the architectural style.
   - The prediction confidence score is displayed.

2. **Heritage Conversational Assistant**
   - Users can ask questions about architectural styles.
   - The system retrieves relevant information from heritage documents.
   - A Large Language Model generates contextual answers with citations.

---

# Features

✅ Image-based architectural style classification  
✅ Retrieval-Augmented Generation (RAG) pipeline  
✅ Vector database search using FAISS  
✅ Heritage document knowledge base  
✅ Conversational memory for multi-turn interactions  
✅ LLM-generated explanations  
✅ Source-based responses  
✅ Streamlit interactive user interface  

---

# Supported Architectural Styles

Dataset source:

https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset

The Computer Vision model currently supports:

- Gothic Architecture
- Tudor Revival Architecture
- Edwardian Architecture
- Georgian Architecture
- Queen Anne Architecture
- Baroque Architecture
- Romanesque Architecture

---

# System Architecture

The system consists of three main components:

## 1. Computer Vision Module

- Deep learning image classification model
- Input: Building image
- Output:
  - Predicted architectural style
  - Confidence score


## 2. RAG Knowledge Retrieval Module

The RAG pipeline uses:

- Heritage JSON documents
- Ollama embeddings
- FAISS vector database

Process:

User Query  
↓  
Convert query into embeddings  
↓  
Search FAISS vector database  
↓  
Retrieve relevant heritage information  
↓  
Send context to LLM  


## 3. Conversational AI Module

The LLM generates responses using:

- Retrieved heritage information
- Previous conversation history
- User questions

---

# Technologies Used

## Programming Language

- Python

## Machine Learning

- TensorFlow / Keras
- Convolutional Neural Network (CNN)

## Generative AI

- LangChain
- Ollama
- LLM models

## Retrieval System

- FAISS
- Vector embeddings

## Application

- Streamlit

## Data Processing

- JSON-based heritage knowledge documents

---

# Project Structure