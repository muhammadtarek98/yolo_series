import torch


def intersection_over_union(predicted, true, box_format='midpoint'):
    global true_y2, pred_x1, true_x1, true_y1, pred_x2, pred_y1, true_x2, pred_y2
    if box_format == "midpoint":
        pred_x1 = predicted[..., 0:1] - predicted[..., 2:3] / 2
        pred_y1 = predicted[..., 1:2] - predicted[..., 3:4] / 2
        pred_x2 = predicted[..., 0:1] + predicted[..., 2:3] / 2
        pred_y2 = predicted[..., 1:2] + predicted[..., 3:4] / 2
        true_x1 = true[..., 0:1] - true[..., 2:3] / 2
        true_y1 = true[..., 1:2] - true[..., 3:4] / 2
        true_x2 = true[..., 0:1] + true[..., 2:3] / 2
        true_y2 = true[..., 1:2] + true[..., 3:4] / 2

    elif box_format == "corners":
        pred_x1 = predicted[..., 0:1]
        pred_y1 = predicted[..., 1:2]
        pred_x2 = predicted[..., 2:3]
        pred_y2 = predicted[..., 3:4]
        true_x1 = true[..., 0:1]
        true_y1 = true[..., 1:2]
        true_x2 = true[..., 2:3]
        true_y2 = true[..., 3:4]

    x1 = torch.max(pred_x1, true_x1)
    y1 = torch.max(pred_y1, true_y1)
    x2 = torch.min(pred_x2, true_x2)
    y2 = torch.min(pred_y2, true_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    true_area = abs((true_x2 - true_x1) * (true_y2 - true_y1))
    union = (pred_area + true_area - intersection + 1e-6)
    iou_results = intersection / union
    return iou_results
