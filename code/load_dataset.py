#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Minimal example: load and inspect the GESD error-slice annotations.

The release contains four annotation files, all in COCO-style JSON
(`images`, `categories`, `annotations`):

    data/coco_detection.json     object detection (COCO images)
    data/kitti_detection.json    object detection (KITTI images)
    data/face_detection.json     face detection
    data/coco_segmentation.json  instance segmentation (COCO images)

Every annotation carries a normalized error-slice label:

    reason       canonical GT error-slice name, or "NA" for a correct instance
    reason_code  the original raw code (kept for traceability)

Detection annotations also have `result` in {TP, FP, FN}, `bbox`, and
`category_id`. Segmentation annotations instead have `inference_segmentation`,
`gt_segmentation`, and `gt_category_id`.

Run from the repository root:  python code/load_dataset.py
"""

import json
from collections import Counter, defaultdict


class GESDDataset:
    """A small, dependency-free loader for the GESD annotation files."""

    def __init__(self, annotation_file):
        with open(annotation_file, "r") as f:
            self.data = json.load(f)
        self.images = {img["id"]: img for img in self.data["images"]}
        self.categories = {cat["id"]: cat for cat in self.data["categories"]}
        self._by_image = defaultdict(list)
        for ann in self.data["annotations"]:
            self._by_image[ann["image_id"]].append(ann)

    def image_ids(self):
        return list(self.images.keys())

    def load_image(self, image_id):
        return self.images.get(image_id)

    def load_annotations(self, image_id):
        """All annotations for one image (raw release schema)."""
        return self._by_image.get(image_id, [])

    def error_slices(self):
        """Annotations that belong to an error slice (reason != 'NA')."""
        return [a for a in self.data["annotations"] if a.get("reason") != "NA"]

    def slice_counts(self):
        """Count of annotations per canonical slice (excluding 'NA')."""
        return Counter(a["reason"] for a in self.error_slices())


if __name__ == "__main__":
    files = [
        "data/coco_detection.json",
        "data/kitti_detection.json",
        "data/face_detection.json",
        "data/coco_segmentation.json",
    ]
    for path in files:
        ds = GESDDataset(path)
        counts = ds.slice_counts()
        print(f"\n=== {path} ===")
        print(f"images: {len(ds.image_ids())}  annotations: {len(ds.data['annotations'])}")
        print(f"error-slice annotations: {sum(counts.values())} across {len(counts)} slices")
        for name, n in counts.most_common(5):
            print(f"  {n:5d}  {name}")

    # Inspect a single error-slice annotation.
    ds = GESDDataset("data/face_detection.json")
    sample = ds.error_slices()[0]
    print("\nExample error-slice annotation:")
    print(json.dumps({k: sample[k] for k in ("id", "image_id", "category_id",
                                             "bbox", "result", "reason_code", "reason")
                      if k in sample}, indent=2))
