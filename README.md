# End-to-End-AI-chatbot
Developed an end-to-end data science pipeline including data preprocessing, exploratory data analysis, rule-based intelligent agent, and comparison of supervised vs. unsupervised machine learning models

## Dataset used
<a href="https://github.com/bharathaigeneralist-droid/End-to-End-AI--chatbot/commit/4896c6fbe88848dedcaeacb7da7310a232bd1da2">Data</a>

# How it works:
chatbot.py — the core engine. On startup, all 100 questions from qa_data.txt are batch-encoded into 384-dimensional vectors using all-MiniLM-L6-v2. At query time, the user's question is encoded and compared against all stored embeddings via cosine similarity (implemented as a dot product since vectors are pre-normalized). The best match above the threshold is returned as the answer.

chatbot_ui.py — a Gradio web interface on top of the engine. It includes a live similarity threshold slider, a confidence score toggle, and a top-k match inspector so you can see which stored questions are being retrieved.

qa_data.txt — 100 Q&A pairs covering Python fundamentals, ML algorithms, deep learning, Transformers, NLP, and the modern AI ecosystem (LangChain, RAG, RLHF, CLIP, diffusion models).
Key design choices:

Pre-normalised embeddings at load time → cosine similarity becomes a single dot product at inference, keeping query latency under 10 ms even for 1000+ Q&A pairs.
Threshold defaulting to 0.50 is a practical starting point; lower it to 0.35 for broader recall, raise to 0.65 for stricter precision.

Swap all-MiniLM-L6-v2 for all-mpnet-base-v2 (768-dim) for higher accuracy at the cost of ~3× more memory.

### install these libraries
- sentence-transformers>=2.7.0
- torch>=2.0.0
- gradio>=4.0.0

- ## Project
<img width="1245" height="656" alt="Screenshot 2026-05-04 195343" src="https://github.com/user-attachments/assets/24814534-72c8-41ea-ace9-2af49c94a7cc" />
