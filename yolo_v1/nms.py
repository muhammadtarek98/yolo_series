import torch
from iou import intersection_over_union


def non_maximum_suppression(preds, iou_threshold, object_probability_threshold, box_format='corners'):
    assert type(preds) == list
    preds = [box for box in preds if box[1] > object_probability_threshold]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    bounding_box_after_nms = []
    while preds:
        chosen_box = preds.pop(0)
        preds = [box for
                        box in preds
                        if (box[0] != chosen_box[0] or
                            intersection_over_union(predicted=torch.tensor(chosen_box[2:]), true=torch.tensor(box[2:]),
                                                    box_format=box_format) < iou_threshold)]
        bounding_box_after_nms.append(chosen_box)

    return bounding_box_after_nms
