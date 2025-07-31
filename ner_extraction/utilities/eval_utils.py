import traceback
from collections import defaultdict
from typing import Any, Optional


def convert_prediction_to_span(text: str, entity: dict[str, Any]) -> Optional[tuple[int, int, str]]:
    """Convert prediction entity to span format.

    Parameters
    ----------
    text : str
        The input text to search for entity.
    entity : dict[str, Any]
        Dictionary containing entity information with 'text' and 'label' keys.

    Returns
    -------
    tuple[int, int, str] | None
        A tuple containing (start_index, end_index, label) if entity is found,
        None if entity text is not found in input text.
    """
    entity_text: str = entity["text"]
    start: int = text.find(entity_text)
    if start == -1:
        return None
    end: int = start + len(entity_text)
    return (start, end, entity["label"])


def convert_predictions_to_spans(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert prediction entities to span format and ensure both lists are sorted by ID.

    Parameters
    ----------
    ground_truth : list[dict[str, Any]]
        List of ground truth dictionaries
    predictions : List[dict[str, Any]]
        List of prediction dictionaries containing entities

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Ground truth and converted predictions sorted by ID.
    """
    pred_converted = []
    for item in predictions:
        if item["id"] is not None:
            spans = []
            for entity in item.get("entities", []):
                span = convert_prediction_to_span(item["text"], entity)
                if span is not None:
                    spans.append(span)
            pred_converted.append({"id": int(item["id"]), "text": item["text"], "label": spans})

    # Sort both lists by id
    gt_sorted = sorted(ground_truth, key=lambda x: x["id"])
    pred_sorted = sorted(pred_converted, key=lambda x: x["id"])
    return gt_sorted, pred_sorted


def evaluate_ner_with_partial(
    gt_data: list[dict], pred_data: list[dict], use_partial_matches: bool = False
) -> dict[str, Any]:
    """
    Evaluate NER performance with exact or partial matching.

    Fixed to correctly align items by ID and handle missing entries.
    """
    metrics = {
        "exact_match": 0,
        "partial_match": 0,
        "missed": 0,
        "wrong_label": 0,
        "spurious": 0,
    }
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    try:
        # Create mappings by ID to handle alignment correctly
        gt_dict = {item["id"]: item for item in gt_data}
        pred_dict = {item["id"]: item for item in pred_data}
        all_ids = set(gt_dict.keys()).union(pred_dict.keys())

        matched_gt_spans = set()
        matched_pred_spans = set()

        for current_id in sorted(all_ids):
            gt_item = gt_dict.get(current_id, {"text": "", "label": []})
            pred_item = pred_dict.get(current_id, {"text": "", "label": []})

            gt_spans = set((start, end, label) for start, end, label in gt_item.get("label", []))  # noqa: C401
            pred_spans = set(  # noqa: C401
                (start, end, label) for start, end, label in pred_item.get("label", [])
            )

            # Process exact matches
            exact_matches = gt_spans & pred_spans
            metrics["exact_match"] += len(exact_matches)
            for span in exact_matches:
                _, _, label = span
                class_metrics[label]["tp"] += 1
                matched_gt_spans.add(span)
                matched_pred_spans.add(span)

            # Track partial matches and wrong labels
            partial_matches = []
            for gt_span in gt_spans - matched_gt_spans:
                gt_start, gt_end, gt_label = gt_span
                found_partial = False
                wrong_label = False

                for pred_span in pred_spans - matched_pred_spans:
                    pred_start, pred_end, pred_label = pred_span

                    overlap_start = max(gt_start, pred_start)
                    overlap_end = min(gt_end, pred_end)
                    if overlap_start >= overlap_end:
                        continue

                    if gt_label == pred_label:
                        partial_matches.append((gt_span, pred_span))
                        metrics["partial_match"] += 1
                        found_partial = True
                        break
                    metrics["wrong_label"] += 1
                    class_metrics[gt_label]["fn"] += 1
                    class_metrics[pred_label]["fp"] += 1
                    matched_gt_spans.add(gt_span)
                    matched_pred_spans.add(pred_span)
                    wrong_label = True
                    break

                if not (found_partial or wrong_label):
                    metrics["missed"] += 1
                    class_metrics[gt_label]["fn"] += 1
                    matched_gt_spans.add(gt_span)

            # Process partial matches based on flag
            for gt_span, pred_span in partial_matches:
                gt_label = gt_span[2]
                pred_label = pred_span[2]
                if use_partial_matches:
                    class_metrics[gt_label]["tp"] += 1
                else:
                    class_metrics[gt_label]["fn"] += 1
                    class_metrics[pred_label]["fp"] += 1
                matched_gt_spans.add(gt_span)
                matched_pred_spans.add(pred_span)

            # Check for spurious predictions (unmatched pred_spans)
            for pred_span in pred_spans - matched_pred_spans:
                metrics["spurious"] += 1
                class_metrics[pred_span[2]]["fp"] += 1
                matched_pred_spans.add(pred_span)

        # Calculate metrics
        total_tp = sum(cm["tp"] for cm in class_metrics.values())
        total_fp = sum(cm["fp"] for cm in class_metrics.values())
        total_fn = sum(cm["fn"] for cm in class_metrics.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        # Per-class metrics
        class_performance = {}
        for label, cm in class_metrics.items():
            tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f = 2 * (p * r) / (p + r) if (p + r) else 0
            class_performance[label] = {
                "precision": round(p, 2),
                "recall": round(r, 2),
                "f1": round(f, 2),
            }

        matching_type = "partial" if use_partial_matches else "exact"
        return {
            "entity_counts": metrics,
            f"micro_precision_{matching_type}": round(precision, 2),
            f"micro_recall_{matching_type}": round(recall, 2),
            f"micro_f1_{matching_type}": round(f1, 2),
            f"class_performance_{matching_type}": class_performance,
        }

    except Exception as e:
        print(f"Error during evaluation: {e}\n{traceback.format_exc()}")
        return {
            "entity_counts": metrics,
            f"micro_precision_{'partial' if use_partial_matches else 'exact'}": 0,
            f"micro_recall_{'partial' if use_partial_matches else 'exact'}": 0,
            f"micro_f1_{'partial' if use_partial_matches else 'exact'}": 0,
            f"class_performance_{'partial' if use_partial_matches else 'exact'}": {},
        }
