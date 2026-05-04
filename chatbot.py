"""
Transformer-based Q&A Chatbot
Uses sentence-transformers for semantic similarity matching.

Install dependencies:
    pip install sentence-transformers torch
"""

import re
import sys
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, util


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "all-MiniLM-L6-v2"   # Fast, 384-dim, ~80 MB
QA_FILE      = "qa_data.txt"         # Q&A dataset file
MIN_SCORE    = 0.50                  # Cosine similarity threshold (0–1)
TOP_K        = 3                     # Return top-k candidates for inspection
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Q&A Loader ───────────────────────────────────────────────────────────────

def load_qa_pairs(filepath: str) -> list[dict]:
    """
    Parse a .txt file with the pattern:
        Q: <question>
        A: <answer>

    Blank lines between pairs are ignored.
    Returns a list of {"question": str, "answer": str} dicts.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Q&A file not found: {filepath}")

    text = path.read_text(encoding="utf-8")
    # Split on double-newlines or by Q:/A: pattern
    blocks = re.split(r"\n(?=Q:)", text.strip())

    qa_pairs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q_match = re.search(r"^Q:\s*(.+?)(?=\nA:)", block, re.DOTALL)
        a_match = re.search(r"^A:\s*(.+)$", block, re.DOTALL | re.MULTILINE)
        if q_match and a_match:
            qa_pairs.append({
                "question": q_match.group(1).strip(),
                "answer":   a_match.group(1).strip()
            })

    print(f"✓ Loaded {len(qa_pairs)} Q&A pairs from '{filepath}'")
    return qa_pairs


# ─── Chatbot Engine ───────────────────────────────────────────────────────────

class TransformerChatbot:
    """
    Semantic Q&A chatbot powered by a Sentence-Transformer encoder.

    Workflow
    --------
    1. Load and encode all questions in the .txt file at startup.
    2. For each user query, encode it and compute cosine similarity
       against all stored question embeddings.
    3. Return the answer of the highest-scoring match if it exceeds
       the configured threshold.
    """

    def __init__(
        self,
        qa_file:    str   = QA_FILE,
        model_name: str   = MODEL_NAME,
        min_score:  float = MIN_SCORE,
        top_k:      int   = TOP_K,
    ) -> None:
        self.min_score = min_score
        self.top_k     = top_k

        print(f"⚙  Loading model '{model_name}' on {DEVICE} …")
        self.model = SentenceTransformer(model_name, device=DEVICE)

        self.qa_pairs  = load_qa_pairs(qa_file)
        self.questions = [pair["question"] for pair in self.qa_pairs]
        self.answers   = [pair["answer"]   for pair in self.qa_pairs]

        print("⚙  Encoding question bank …")
        self.question_embeddings = self.model.encode(
            self.questions,
            convert_to_tensor=True,
            device=DEVICE,
            show_progress_bar=True,
            normalize_embeddings=True,   # pre-normalise for fast dot-product
        )
        print("✓ Chatbot is ready!\n")

    # ── Core retrieval ────────────────────────────────────────────────────────

    def retrieve(self, user_query: str) -> list[dict]:
        """
        Encode the query and return the top-k matches with scores.

        Returns
        -------
        list of dicts: [{"question", "answer", "score"}, …]
        """
        query_embedding = self.model.encode(
            user_query,
            convert_to_tensor=True,
            device=DEVICE,
            normalize_embeddings=True,
        )

        # Cosine similarity (dot-product since vectors are pre-normalised)
        scores = util.dot_score(query_embedding, self.question_embeddings)[0]
        top_results = torch.topk(scores, k=min(self.top_k, len(self.questions)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "question": self.questions[idx],
                "answer":   self.answers[idx],
                "score":    float(score),
            })
        return results

    def respond(self, user_query: str) -> tuple[str, float]:
        """
        Return (answer_text, confidence_score) for a user query.
        Falls back to a polite message if confidence is below threshold.
        """
        user_query = user_query.strip()
        if not user_query:
            return "Please type a question.", 0.0

        results = self.retrieve(user_query)
        best    = results[0] if results else None

        if best and best["score"] >= self.min_score:
            return best["answer"], best["score"]

        score = best["score"] if best else 0.0
        return (
            "I'm sorry, I don't have a confident answer for that question. "
            "Try rephrasing or ask about Python, ML, deep learning, or NLP topics.",
            score,
        )

    # ── CLI interaction ───────────────────────────────────────────────────────

    def chat(self, show_score: bool = True, show_top_k: bool = False) -> None:
        """
        Interactive command-line chat loop.

        Commands
        --------
        quit / exit / q  – end the session
        /top             – toggle display of top-k matches
        /score           – toggle confidence score display
        /reload          – reload and re-encode the Q&A file
        """
        print("=" * 60)
        print("  Transformer Q&A Chatbot")
        print("  Type 'quit' to exit | '/top' to see top matches")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            lower = user_input.lower()
            if lower in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if lower == "/top":
                show_top_k = not show_top_k
                print(f"[Top-{self.top_k} display: {'ON' if show_top_k else 'OFF'}]")
                continue
            if lower == "/score":
                show_score = not show_score
                print(f"[Score display: {'ON' if show_score else 'OFF'}]")
                continue
            if lower == "/reload":
                self.qa_pairs  = load_qa_pairs(QA_FILE)
                self.questions = [p["question"] for p in self.qa_pairs]
                self.answers   = [p["answer"]   for p in self.qa_pairs]
                self.question_embeddings = self.model.encode(
                    self.questions, convert_to_tensor=True,
                    device=DEVICE, normalize_embeddings=True,
                )
                print("[Q&A database reloaded]")
                continue

            # Retrieve and display
            if show_top_k:
                results = self.retrieve(user_input)
                print(f"\n── Top {self.top_k} matches ──────────────────────")
                for i, r in enumerate(results, 1):
                    print(f"  {i}. [{r['score']:.3f}] Q: {r['question'][:70]}")
                print()

            answer, score = self.respond(user_input)
            score_tag = f"  [confidence: {score:.3f}]" if show_score else ""
            print(f"Bot: {answer}{score_tag}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transformer Q&A Chatbot")
    parser.add_argument("--qa-file",    default=QA_FILE,    help="Path to Q&A .txt file")
    parser.add_argument("--model",      default=MODEL_NAME, help="Sentence-Transformer model name")
    parser.add_argument("--threshold",  default=MIN_SCORE,  type=float, help="Minimum similarity score")
    parser.add_argument("--top-k",      default=TOP_K,      type=int,   help="Number of top matches to retrieve")
    parser.add_argument("--no-score",   action="store_true",            help="Hide confidence scores")
    parser.add_argument("--query",      default=None,                   help="Single query mode (non-interactive)")
    args = parser.parse_args()

    bot = TransformerChatbot(
        qa_file=args.qa_file,
        model_name=args.model,
        min_score=args.threshold,
        top_k=args.top_k,
    )

    if args.query:
        answer, score = bot.respond(args.query)
        print(f"Q: {args.query}")
        print(f"A: {answer}")
        if not args.no_score:
            print(f"Confidence: {score:.3f}")
    else:
        bot.chat(show_score=not args.no_score)


if __name__ == "__main__":
    main()
