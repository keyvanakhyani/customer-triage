"""Evaluate classification accuracy against a labeled test set."""

import json

from core.triage import build_classify_chain


def run_evaluation():
    """Run the classifier on the test set and report accuracy."""
    with open("eval_data.json", encoding="utf-8") as f:
        test_cases = json.load(f)

    chain = build_classify_chain()
    correct = 0

    for case in test_cases:
        predicted = chain.invoke({"message": case["message"]}).strip()
        expected = case["category"]
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"{mark} expected={expected:20} predicted={predicted}")

    total = len(test_cases)
    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")


if __name__ == "__main__":
    run_evaluation()