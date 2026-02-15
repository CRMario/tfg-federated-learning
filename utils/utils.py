import numpy as np
from typing import cast
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    NDArray,
    RecordDict,
    log,
)

def aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
        ) -> MetricRecord:
        """Perform weighted aggregation all MetricRecords using a specific key."""
        # Retrieve weighting factor from MetricRecord
        weights: list[float] = []
        for record in records:
            # Get the first (and only) MetricRecord in the record
            metricrecord = next(iter(record.metric_records.values()))
            # Because replies have been checked for consistency,
            # we can safely cast the weighting factor to float
            w = cast(float, metricrecord[weighting_metric_name])
            weights.append(w)

        # Average
        total_weight = sum(weights)
        weight_factors = [w / total_weight for w in weights]

        aggregated_metrics = MetricRecord()
        aggregated_cm = None
        for record, weight in zip(records, weight_factors, strict=True):
            for record_item in record.metric_records.values():
                # aggregate in-place
                for key, value in record_item.items():
                    if key == "confusion_matrix":
                        # value: len(labels)*len(labels) confusion matrix
                        side_length = int(len(value)**0.5) # we had to flatten it to send it through ray
                        matrix = np.array(value).reshape((side_length, side_length)) # since its square we can reshape it back
                        if aggregated_cm is None: #initialize the aggregated confusion matrix
                            # aggregated_cm[i][0]: true positives for the label i
                            # aggregated_cm[i][1]: false positives for the label i 
                            # aggregated_cm[i][2]: false negatives for the label i 
                            aggregated_cm = {i: [0,0,0] for i in range(matrix.shape[1])}
                        for i in range(matrix.shape[1]):
                            col_values = np.sum(matrix[:,i]) # all values of the column label
                            row_values = np.sum(matrix[i, :])
                            tps = matrix[i,i] # true positives of the class is the value in the diagonal of the matrix for that column
                            aggregated_cm[i][0] += tps
                            aggregated_cm[i][1] += col_values - tps
                            aggregated_cm[i][2] += row_values - tps
                        continue
                    if key == weighting_metric_name:
                        # We exclude the weighting key from the aggregated MetricRecord
                        continue
                    if key not in aggregated_metrics:
                        if isinstance(value, list):
                            aggregated_metrics[key] = [v * weight for v in value]
                        else:
                            aggregated_metrics[key] = value * weight
                    else:
                        if isinstance(value, list):
                            current_list = cast(list[float], aggregated_metrics[key])
                            aggregated_metrics[key] = [
                                curr + val * weight
                                for curr, val in zip(current_list, value, strict=True)
                            ]
                        else:
                            current_value = cast(float, aggregated_metrics[key])
                            aggregated_metrics[key] = current_value + value * weight

        # measure precision for each label based on the confusion matrix
        for label_id, values in aggregated_cm.items():
            tp, fp, fn = values
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            aggregated_metrics[f"precision_label{label_id}"] = precision.item()
            aggregated_metrics[f"recall_label{label_id}"] = recall.item()
            aggregated_metrics[f"f1_score_label{label_id}"] = f1_score.item()

        return aggregated_metrics
        