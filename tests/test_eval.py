from drive_vlm.eval import accuracy_at_50, iou, parse_box


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


# --- parse_box: the box-as-text parser shared by the Qwen eval/train/demo paths ---


def test_parse_box_bracketed():
    assert parse_box("[975, 463, 1141, 575]") == (975.0, 463.0, 1141.0, 575.0)


def test_parse_box_floats():
    assert parse_box("100.5 200 300.25 400") == (100.5, 200.0, 300.25, 400.0)


def test_parse_box_ignores_surrounding_text():
    assert parse_box("The referred object is at [10, 20, 30, 40].") == (10.0, 20.0, 30.0, 40.0)


def test_parse_box_negative_coords():
    # findall keeps the sign — a box clipped past the left/top edge stays intact.
    assert parse_box("[-5, 0, 10, 20]") == (-5.0, 0.0, 10.0, 20.0)


def test_parse_box_takes_first_four():
    assert parse_box("[1, 2, 3, 4, 5, 6]") == (1.0, 2.0, 3.0, 4.0)


def test_parse_box_too_few_numbers_is_none():
    assert parse_box("only two numbers: 10, 20") is None


def test_parse_box_no_numbers_is_none():
    assert parse_box("I could not find the object.") is None
