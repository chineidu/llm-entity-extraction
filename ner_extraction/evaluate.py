import traceback
from pathlib import Path
from typing import Any

from rich.pretty import pprint
from utilities.eval_utils import (  # type: ignore
    convert_predictions_to_spans,
    evaluate_ner_with_partial,
)
from utilities.utils import load_json_data  # type: ignore

from ner_extraction import PROJECT_ROOT, create_logger

logger = create_logger(name="evaluate")


def perform_evaluation(
    ground_truth: list[dict[str, str]],
    pred_with_span: list[dict[str, str]],
    num_examples: int = 5,
    convert_to_spans: bool = True,
    use_partial_matches: bool = True,
) -> dict[str, Any]:
    """Evaluate and print Named Entity Recognition (NER) performance metrics.

    Parameters
    ----------
    ground_truth : list[dict[str, str]]
        List of dictionaries containing ground truth annotations with keys for 'id',
        'text', and 'label'.
    pred_with_span : list[dict[str, str]]
        List of dictionaries containing model predictions with keys for 'id',
        'text', and 'label'.
    num_examples: int, default=5
        Number of examples to display the error analysis.
    convert_to_spans : bool, optional
        If True (default), convert ground truth and predictions to entity spans.
    use_partial_matches : bool, optional
        If True (default), count partial matches as true positives when calculating metrics.
        If False, only count exact matches as true positives.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation results.
    """
    logger.info("Running evaluation...")
    matching_type: str = "partial" if use_partial_matches else "exact"
    try:
        if convert_to_spans:
            ground_truth, pred_with_span = convert_predictions_to_spans(
                ground_truth, pred_with_span
            )

        # Evaluate entity recognition performance
        results: dict[str, Any] = evaluate_ner_with_partial(
            ground_truth, pred_with_span, use_partial_matches=use_partial_matches
        )

        # Print overall metrics
        pprint("Entity Recognition Performance:")
        print(f"Entity counts: {results['entity_counts']}")
        print(f"Micro precision [{matching_type}]: {results[f'micro_precision_{matching_type}']}")
        print(f"Micro recall [{matching_type}]: {results[f'micro_recall_{matching_type}']}")
        print(f"Micro F1 [{matching_type}]: {results[f'micro_f1_{matching_type}']}")
        print(
            f"Class Performance [{matching_type}]: {results[f'class_performance_{matching_type}']}"
        )

        # Print per-class metrics
        pprint("\nPer-Class Performance:")
        for label, metrics in results[f"class_performance_{matching_type}"].items():
            print(f"{label}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}")

        # Error analysis - examine specific examples
        pprint("\nError Analysis:")
        for gt, pred in zip(
            ground_truth[:num_examples], pred_with_span[:num_examples], strict=True
        ):
            print(f"Example {gt['id']}:")
            print(f"Text: {gt['text']}")
            print(f"Ground Truth: {gt['label']}")
            print(f"Prediction: {pred['label']}\n")

        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}


if __name__ == "__main__":
    logger.info("Loading data...")
    fp_1: Path = PROJECT_ROOT / "data/results/combined/ground_truth.jsonl"
    fp_2: Path = PROJECT_ROOT / "data/results/combined/results_GEMINI_2p5_FLASH_REMOTE.json"
    fp_3: Path = PROJECT_ROOT / "data/results/combined/preds_GLiNER.jsonl"
    true_data: list[dict[str, str]] = load_json_data(fp_1, format_type="jsonl_file")  # [:10]
    predictions: list[dict[str, str]] = load_json_data(fp_2)  # [:10]
    # predictions: list[dict[str, str]] = load_json_data(fp_3, format_type="jsonl_file")

    perform_evaluation(
        ground_truth=true_data,
        pred_with_span=predictions,
        convert_to_spans=True,
        use_partial_matches=True,
    )
