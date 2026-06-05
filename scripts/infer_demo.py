"""Quick manual check: run Grounding DINO on one image + command.

    uv run python scripts/infer_demo.py path/to/image.jpg "the white truck"
"""

import sys

from PIL import Image

from drive_vlm.model import GroundingDINO


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    image_path, command = sys.argv[1], sys.argv[2]

    model = GroundingDINO()
    box = model.predict(Image.open(image_path).convert("RGB"), command)
    print(f"command: {command!r}")
    print(f"box: {box}")


if __name__ == "__main__":
    main()
