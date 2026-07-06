import json

from drive_vlm.data import load_split, xywh_to_xyxy


def test_xywh_to_xyxy_converts_topleft_wh_to_corners():
    # Talk2Car's 2d_box is [x, y, w, h] — the bug that once silently corrupted the metric.
    assert xywh_to_xyxy([100, 50, 200, 150]) == (100, 50, 300, 200)


def test_xywh_to_xyxy_zero_size():
    assert xywh_to_xyxy([5, 5, 0, 0]) == (5, 5, 5, 5)


def test_load_split_reads_and_converts(tmp_path):
    (tmp_path / "images").mkdir()
    payload = {
        "commands": [
            {
                "t2c_img": "img_val_0.jpg",
                "command": "the white truck",
                "2d_box": [100, 50, 200, 150],  # xywh
                "obj_name": "vehicle.truck",
            }
        ]
    }
    (tmp_path / "val_commands.json").write_text(json.dumps(payload))

    samples = load_split(tmp_path, "val")

    assert len(samples) == 1
    s = samples[0]
    assert s.command == "the white truck"
    assert s.label == "vehicle.truck"
    assert s.box == (100, 50, 300, 200)  # converted xywh -> xyxy on load
    assert s.image_path == tmp_path / "images" / "img_val_0.jpg"
