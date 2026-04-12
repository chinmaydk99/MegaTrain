from infinity.evaluation import (
    build_dataset_splits,
    exact_match_score,
    extract_final_answer,
    normalize_math_answer,
    split_messages_for_evaluation,
)


def test_build_dataset_splits_is_deterministic():
    first = build_dataset_splits(10, train_ratio=0.7, eval_ratio=0.3, seed=42)
    second = build_dataset_splits(10, train_ratio=0.7, eval_ratio=0.3, seed=42)
    assert first == second


def test_build_dataset_splits_has_no_overlap():
    train_indices, eval_indices = build_dataset_splits(10, train_ratio=0.7, eval_ratio=0.3, seed=7)
    assert len(train_indices) == 7
    assert len(eval_indices) == 3
    assert set(train_indices).isdisjoint(eval_indices)


def test_extract_final_answer_handles_common_metamath_formats():
    boxed = (
        "Reasoning...\n"
        "Therefore, Gracie and Joe's points are \\boxed{\\sqrt{5}} units apart.\n"
        "The answer is: \\sqrt{5}"
    )
    hashes = "Compute carefully.\n#### 752"
    plain = "The answer is: 17/12"

    assert extract_final_answer(boxed) == "\\sqrt{5}"
    assert extract_final_answer(hashes) == "752"
    assert extract_final_answer(plain) == "17/12"


def test_exact_match_score_normalizes_minor_formatting():
    assert normalize_math_answer("$\\sqrt{5}$") == "\\sqrt{5}"
    assert exact_match_score("The answer is: 752.", "#### 752")
    assert exact_match_score("Reasoning... \\boxed{\\frac{3}{4}}", "The answer is: \\frac{3}{4}")


def test_split_messages_for_evaluation_uses_last_assistant_turn():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Intermediate answer"},
        {"role": "user", "content": "Final question"},
        {"role": "assistant", "content": "Final answer"},
    ]

    prompt_messages, reference = split_messages_for_evaluation(messages)

    assert reference == "Final answer"
    assert prompt_messages == messages[:-1]
