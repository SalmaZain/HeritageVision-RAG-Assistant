import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import ollama

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="UK Architectural Style Classification",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ UK Architectural Style Classification")
st.write(
    "Upload an image of a building to classify its architectural style "
    "and receive an AI-generated explanation."
)

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = (224, 224)

CLASS_LABELS = [
    "baroque_architecture",
    "edwardian_architecture",
    "georgian_architecture",
    "gothic_architecture",
    "queen_anne_architecture",
    "romanesque_architecture",
    "tudor_revival_architecture"
]

# -----------------------------
# Load CNN Model
# -----------------------------
@st.cache_resource
def load_cnn():
    model = load_model("nhpt_model.h5")
    return model

model = load_cnn()

# -----------------------------
# Load FAISS
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
	base_url="http://127.0.0.1:11434"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

vectorstore = load_vectorstore()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(uploaded_image):

    image = uploaded_image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = preprocess_input(img.astype(np.float32))

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Prediction
# -----------------------------
def predict(image):

    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)[0]

    class_index = np.argmax(prediction)

    class_name = CLASS_LABELS[class_index]

    confidence = float(prediction[class_index])

    return class_name, confidence

# -----------------------------
# RAG + Ollama Explanation
# -----------------------------
def get_architecture_explanation(style_name):

    query = f"Explain {style_name.replace('_', ' ')} architecture."

    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join(
        f"[{doc.metadata.get('source','Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""
You are an expert in British architectural heritage.

Use ONLY the provided context to answer.

Context:
{context}

Question:
Explain the architectural style "{style_name.replace('_',' ')}".

Include:
- Historical period
- Main characteristics
- Common materials
- Famous examples
- Interesting facts

Answer in clear paragraphs.
"""

    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    sources = [
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]

    return answer, sources


# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
        )

    with col2:

        with st.spinner("Predicting architectural style..."):

            prediction, confidence = predict(image)

        st.success("Prediction Complete")

        st.subheader("Predicted Style")

        st.write(
            prediction.replace("_", " ").title()
        )

        st.progress(float(confidence))

        st.write(
            f"Confidence: **{confidence*100:.2f}%**"
        )

    st.divider()

    st.subheader("AI Heritage Explanation")

    with st.spinner("Generating explanation..."):

        explanation, sources = get_architecture_explanation(
            prediction
        )

    st.write(explanation)

    st.subheader("Knowledge Sources")

    for source in sources:
        st.write("•", source)

# ======================================
# Session State
# ======================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ======================================
# Sidebar
# ======================================
with st.sidebar:

    st.title("🏛️ About")

    st.write("""
This application classifies UK architectural styles using an EfficientNetB0 deep learning model.

After prediction, a Retrieval-Augmented Generation (RAG) system retrieves relevant heritage documents from a FAISS vector database and Gemma 3 generates an explanation.

**Model**
- EfficientNetB0
- 7 Classes

**LLM**
- Gemma 3 12B

**Vector Database**
- FAISS

**Embedding Model**
- nomic-embed-text
""")

    st.divider()

    st.write("Developed using")

    st.write("• TensorFlow")
    st.write("• Streamlit")
    st.write("• LangChain")
    st.write("• Ollama")
    st.write("• FAISS")


# ======================================
# Chat Assistant
# ======================================

st.divider()

st.header("💬 Ask the Heritage Assistant")

question = st.text_input(
    "Ask any question about British architecture..."
)

if st.button("Ask"):

    if question.strip() != "":

        with st.spinner("Searching knowledge base..."):

            docs = vectorstore.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                f"[{doc.metadata.get('source','Unknown')}]\n{doc.page_content}"
                for doc in docs
            )

            prompt = f"""
You are an expert in British architectural heritage.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""

            response = ollama.chat(
                model="gemma3:12b",
                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            st.session_state.chat_history.append(
                ("You", question)
            )

            st.session_state.chat_history.append(
                ("Assistant", answer)
            )


# ======================================
# Display Chat
# ======================================

if len(st.session_state.chat_history) > 0:

    st.subheader("Conversation")

    for sender, message in st.session_state.chat_history:

        if sender == "You":

            st.markdown(
                f"""
<div style="background:#DCF8C6;
padding:10px;
border-radius:10px;
margin-bottom:8px">
<b>You</b><br>
{message}
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div style="background:#F3F3F3;
padding:10px;
border-radius:10px;
margin-bottom:12px">
<b>Assistant</b><br>
{message}
</div>
""",
                unsafe_allow_html=True
            )


# ======================================
# Footer
# ======================================

st.divider()

st.caption(
    "UK Architectural Style Classification using EfficientNetB0 + FAISS + Ollama (Gemma 3)"
)