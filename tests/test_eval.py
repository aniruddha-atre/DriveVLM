from drive_vlm.eval import accuracy_at_50, iou


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


def test_accuracy_at_50_counts_hits_and_misses():
    gts = [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (8, 8, 18, 18), None]  # hit, miss (low IoU), miss (None)
    assert accuracy_at_50(preds, gts) == 1 / 3
