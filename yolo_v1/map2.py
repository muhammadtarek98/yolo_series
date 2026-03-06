import numpy as np


def calculate_precision(tp, fp):
    return tp / (tp + fp)


def calculate_average_precision(precision, recall):
    # Calculate the area under the precision-recall curve using the trapezoidal rule
    return np.trapz(precision, recall)


def calculate_mAP(predictions, ground_truth):
    num_classes = len(predictions)
    ap_values = []

    for class_idx in range(num_classes):
        class_predictions = predictions[class_idx]
        class_ground_truth = ground_truth[class_idx]

        sorted_indices = np.argsort(class_predictions)[::-1]
        sorted_ground_truth = class_ground_truth[sorted_indices]

        tp = 0
        fp = 0
        precision = []
        recall = []

        for i in range(len(sorted_ground_truth)):
            if sorted_ground_truth[i] == 1:
                tp += 1
            else:
                fp += 1

            precision.append(calculate_precision(tp, fp))
            recall.append(tp / np.sum(class_ground_truth))

        ap = calculate_average_precision(precision, recall)
        ap_values.append(ap)

    mAP = np.mean(ap_values)
    return mAP
