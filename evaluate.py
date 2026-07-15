"""Evaluate classification accuracy against a labeled test set."""

import json

from core.triage import build_classify_chain_raw


def run_evaluation():
    """Run the classifier on the test set and report accuracy per case."""
    with open("eval_data.json", encoding="utf-8") as f:
        test_cases = json.load(f)

    chain = build_classify_chain_raw()
    correct = 0
    tested = 0
    models_seen = set()

    for index, case in enumerate(test_cases, start=1):
        case_id = case.get("id", index)

        try:
            response = chain.invoke({"message": case["message"]})
        except Exception as error:
            print(f"#{case_id:2} ⚠ skipped: {error}")
            continue

        predicted = response.content.strip()
        model_used = response.response_metadata.get("model_name", "unknown")
        models_seen.add(model_used)

        tested += 1
        expected = case["category"]
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"#{case_id:2} {mark} expected={expected:18} predicted={predicted}")

    accuracy = correct / tested * 100 if tested else 0
    print(f"\nTested: {tested}/{len(test_cases)}")
    print(f"Accuracy: {correct}/{tested} = {accuracy:.1f}%")
    print(f"Models used: {', '.join(models_seen)}")


if __name__ == "__main__":
    run_evaluation()