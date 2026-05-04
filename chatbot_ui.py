"""
Gradio Web UI for the Transformer Q&A Chatbot.

Install dependencies:
    pip install sentence-transformers torch gradio

Run:
    python chatbot_ui.py
"""

import gradio as gr
from chatbot import TransformerChatbot, QA_FILE, MODEL_NAME, MIN_SCORE, TOP_K


# ─── Initialise once at startup ───────────────────────────────────────────────

print("Initialising chatbot …")
bot = TransformerChatbot(
    qa_file=QA_FILE,
    model_name=MODEL_NAME,
    min_score=MIN_SCORE,
    top_k=TOP_K,
)


# ─── Gradio handler functions ─────────────────────────────────────────────────

def chat_fn(
    user_message: str,
    history:      list[list],
    threshold:    float,
    show_score:   bool,
    show_top_k:   bool,
) -> tuple[str, list[list], str]:
    """
    Called on every user submission.

    Returns
    -------
    ("", updated_history, top_k_markdown)
    """
    bot.min_score = threshold   # honour live slider
    answer, score = bot.respond(user_message)

    display = answer
    if show_score:
        display += f"\n\n*Confidence: {score:.3f}*"

    history.append([user_message, display])

    # Build top-k table
    top_k_md = ""
    if show_top_k:
        results = bot.retrieve(user_message)
        rows = [f"| {r['score']:.3f} | {r['question']} |" for r in results]
        top_k_md = (
            f"### Top {bot.top_k} Matches\n"
            "| Score | Matched Question |\n"
            "|-------|------------------|\n"
            + "\n".join(rows)
        )

    return "", history, top_k_md


def clear_fn() -> tuple[list, str]:
    return [], ""


# ─── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Transformer Q&A Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Transformer Q&A Chatbot
        Powered by **Sentence-Transformers** (`all-MiniLM-L6-v2`) and cosine similarity.
        Ask anything about **Python · Machine Learning · Deep Learning · NLP**.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                label="Conversation",
                height=480,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about Python, ML, or AI …",
                    label="Your question",
                    scale=5,
                    lines=1,
                )
                send_btn  = gr.Button("Send",  variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            top_k_output = gr.Markdown(label="Top matches")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            threshold_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05,
                value=MIN_SCORE,
                label="Similarity threshold",
                info="Minimum cosine score to return a match",
            )
            show_score_chk = gr.Checkbox(value=True,  label="Show confidence score")
            show_top_k_chk = gr.Checkbox(value=False, label="Show top-k matches")

            gr.Markdown("---")
            gr.Markdown(
                f"**Model:** `{MODEL_NAME}`  \n"
                f"**Q&A pairs loaded:** {len(bot.qa_pairs)}  \n"
                f"**Embedding dim:** 384  \n"
                f"**Device:** `{bot.model.device}`"
            )

            gr.Markdown("---")
            gr.Markdown(
                "### Example questions\n"
                "- What is a Transformer model?\n"
                "- How does gradient descent work?\n"
                "- What is BERT?\n"
                "- Explain cosine similarity\n"
                "- What is RAG?\n"
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    shared_inputs  = [msg_box, chatbot_ui, threshold_slider, show_score_chk, show_top_k_chk]
    shared_outputs = [msg_box, chatbot_ui, top_k_output]

    send_btn.click(chat_fn, shared_inputs, shared_outputs)
    msg_box.submit(chat_fn, shared_inputs, shared_outputs)
    clear_btn.click(clear_fn, outputs=[chatbot_ui, top_k_output])


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public Gradio link
        inbrowser=True,
    )
