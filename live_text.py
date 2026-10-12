"""
Standalone Native Real-Time Text Emotion Recognition Program.
Interactive terminal interface classifying typed sentences using the trained DistilRoBERTa model.
Computes genuine 7-class emotion probabilities and circumplex valence/arousal coordinates.
"""

import os
import sys
import argparse
import logging

# Suppress verbose huggingface/transformers logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

from core.text.model import TextEmotionModel
from core.ontology import CORE_EMOTIONS


def clear_terminal():
    """Clear terminal screen cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    """Display interactive application banner."""
    print("=" * 65)
    print("       LIVE TEXT EMOTION RECOGNITION (DistilRoBERTa)")
    print("=" * 65)
    print("  Model:            j-hartmann/emotion-english-distilroberta-base")
    print("  Ontology:         Ekman 7-Class Canonical Representation")
    print("  Inference:        Direct PyTorch Transformer Forward Pass")
    print("  Instructions:     Type a sentence and press ENTER.")
    print("  Commands:         'exit' or 'quit' to terminate | 'clear' to clear screen")
    print("=" * 65 + "\n")


def format_probability_bar(p: float, width: int = 24) -> str:
    """Format an ASCII progress bar for probability display."""
    filled = int(round(p * width))
    bar = "#" * filled + " " * (width - filled)
    return f"[{bar}]"


def run_interactive_text(model_name: str = "j-hartmann/emotion-english-distilroberta-base", device: str = "cpu"):
    """Interactive REPL loop for text emotion classification."""
    print("\n[INFO] Loading RoBERTa text emotion model...")
    try:
        model = TextEmotionModel(model_name=model_name, device=device)
    except Exception as e:
        print(f"[FATAL] Failed to load text model: {e}")
        sys.exit(1)

    if not model._is_loaded:
        print(f"[FATAL] Text model failed to initialize: {model.load_error}")
        sys.exit(1)

    clear_terminal()
    print_banner()

    query_count = 0

    while True:
        try:
            user_input = input("Text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[LIVE TEXT] Interrupted by user. Exiting cleanly.\n")
            break

        if not user_input:
            continue

        lower_cmd = user_input.lower()
        if lower_cmd in ("exit", "quit", ":q"):
            print("\n[LIVE TEXT] Exiting program. Goodbye!\n")
            break

        if lower_cmd == "clear":
            clear_terminal()
            print_banner()
            continue

        query_count += 1

        # Run genuine transformer inference
        try:
            result = model.predict(user_input)
        except Exception as e:
            print(f"\n[ERROR] Inference error: {e}\n")
            continue

        if not result.available:
            print(f"\n[ERROR] Text analysis unavailable: {result.status}\n")
            continue

        emotion = result.emotion.upper()
        confidence = result.confidence * 100.0
        probs = result.probabilities

        # Affective state descriptor
        v = result.valence
        a = result.arousal
        v_label = "Positive affect" if v >= 0 else "Negative affect"
        a_label = "High intensity" if a >= 0 else "Calm / Low intensity"

        print("\n" + "-" * 65)
        print(f"  Emotion:    {emotion}")
        print(f"  Confidence: {confidence:.2f}%  (Model Certainty Score - Not Test Accuracy)")
        print(f"  Circumplex: Valence: {v:+.2f} ({v_label}) | Arousal: {a:+.2f} ({a_label})")
        print("-" * 65)
        print("  7-Class Probability Distribution:")

        for em in CORE_EMOTIONS:
            p = probs.get(em, 0.0)
            is_dominant = (em == result.emotion)
            prefix = "-> " if is_dominant else "   "
            bar = format_probability_bar(p, width=22)
            print(f"{prefix}{em:<9} {p * 100:6.2f}%  {bar}")

        print("-" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Interactive Text Emotion Recognition using DistilRoBERTa")
    parser.add_argument("--model", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="Model name or path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    run_interactive_text(model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
