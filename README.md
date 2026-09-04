# GESD: Grounded Error Slice Dataset

This repository contains the official dataset release accompanying the ECCV 2026 paper
**[GH-ESD: Grounded Hypothesis-Driven Error Slice Discovery for Instance-Level Vision Tasks](https://arxiv.org/abs/2512.24592)**.

GESD is a benchmark for fine-grained, grounded error-slice discovery on
instance-level vision tasks (object detection and instance segmentation). See
the paper for the GH-ESD method and how GESD was built.

## The GESD Dataset

GESD covers 12,116 images from the validation splits of three public datasets.
Error slices were defined by expert hypotheses and matched to instances through
human annotation, with corrected ground-truth boxes/masks. Each slice is grounded
to a local error region. The benchmark defines **21 object-detection slices** and
**21 instance-segmentation slices**.

## Files

All annotations are COCO-style JSON (`images`, `categories`, `annotations`).

| File | Task | Images |
|---|---|---|
| `data/coco_detection.json` | Object detection (COCO) | 5,000 |
| `data/kitti_detection.json` | Object detection (KITTI) | 3,769 |
| `data/face_detection.json` | Face detection | 3,347 |
| `data/coco_segmentation.json` | Instance segmentation (COCO) | 5,000 |

The two COCO files annotate the same 5,000 images. The three detection files
share one category taxonomy and together form the detection benchmark, where a
slice such as *Blurry face missed* may appear in more than one file.

## Annotation Format

Every annotation includes a normalized error-slice label:

| Field | Meaning |
|---|---|
| `reason` | Canonical ground-truth slice name, or `"NA"` for a correct (non-error) instance |
| `reason_code` | The original raw code for error instances; `"NA"` for correct instances |

**Detection** annotations also carry `result` ∈ {`TP`, `FP`, `FN`}, `bbox`, and
`category_id`. **Segmentation** annotations instead carry
`inference_segmentation`, `gt_segmentation`, and `gt_category_id`; both
`category_id` and `gt_category_id` index the file's `categories` array.

Correct instances (`result == "TP"`, and the clean ground-truth instances in the
segmentation file) are unified under `reason == "NA"`. The complete mapping from
raw codes to the 21 + 21 canonical slice names is documented in
[`docs/slice-mapping.md`](docs/slice-mapping.md).

Example detection annotation:

```json
{
  "id": 2,
  "image_id": 1,
  "category_id": 3,
  "bbox": [45, 182, 55, 53],
  "result": "FN",
  "reason_code": "blurry_face",
  "reason": "Blurry face missed"
}
```

## Getting Started

`code/load_dataset.py` is a small, dependency-free example that loads any of the
four annotation files and reports per-slice statistics. Run the examples from the
repository root:

```bash
python code/load_dataset.py
```

```python
import sys; sys.path.insert(0, "code")
from load_dataset import GESDDataset

ds = GESDDataset("data/coco_segmentation.json")
print(ds.slice_counts())          # annotations per canonical slice
slices = ds.error_slices()        # annotations with reason != "NA"
```

## Evaluating Precision@k

`code/evaluate_precision.py` computes **Precision@k** for your discovered error
slices against the GESD ground truth.

Your prediction file must be a JSON dictionary where:
- **Keys**: Any string representing the predicted slice (e.g., `"pred_slice_1"`, `"1"`, or a semantic description like `"blurry faces"`).
- **Values**: A list of string IDs representing the instances in that slice, sorted by confidence (highest confidence first). Because annotation IDs reset to 1 in each JSON file, each ID must be prefixed with the dataset name, formatted as `{dataset_name}_{annotation_id}` (e.g., `"coco_detection_1024"`, `"face_detection_15"`).

Example of a valid prediction format:

```json
{
    "blurry faces": ["coco_detection_1024", "kitti_detection_2048", "face_detection_512"],
    "1": ["coco_detection_99", "coco_detection_100"]
}
```

Evaluate the **Detection** task (the three detection files are merged into one
ground truth):
```bash
python code/evaluate_precision.py --task detection --pred_file path/to/det_preds.json --k 10
```

Evaluate the **Segmentation** task:
```bash
python code/evaluate_precision.py --task segmentation --pred_file path/to/seg_preds.json --k 10
```

## License

The sample code in this repository (`code/load_dataset.py`,
`code/evaluate_precision.py`) is released under the
[Apple Sample Code License](code/LICENSE).

The annotation files in `data/` are licensed separately, by source dataset:

| File | Derived from | License |
| --- | --- | --- |
| `data/coco_detection.json`, `data/coco_segmentation.json` | COCO | [Apple Sample Code License](data/LICENSE-APPLE-SAMPLE-CODE) |
| `data/face_detection.json` | Public Face Detection Dataset | [Apple Sample Code License](data/LICENSE-APPLE-SAMPLE-CODE) |
| `data/kitti_detection.json` | KITTI | [CC BY-NC-SA 3.0](data/LICENSE-CC-BY-NC-SA-3.0) |

`data/kitti_detection.json` is a derivative of the KITTI 2D object dataset, so
under the ShareAlike terms it inherits KITTI's license and is for
**non-commercial use only**; the other files are unaffected.

### Source datasets and underlying images

This repository distributes **error-slice annotations only**. Users must obtain
the images from the original providers under their respective terms:

* **COCO** — https://cocodataset.org/#download. The images are sourced from
  Flickr and their copyrights remain with the respective copyright holders under
  the licenses associated with the original Flickr content.
* **KITTI** — https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
* **Public Face Detection Dataset** —
  https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset/data
  The underlying images are sourced from **Google Open Images**
  (https://storage.googleapis.com/openimages/web/index.html), where they are
  listed as having a **CC BY 2.0** license.

The images are copyright of their respective owners; per-image license terms can
be resolved as follows:

* In `data/coco_detection.json` and `data/coco_segmentation.json`, every entry
  in `images` carries the original COCO `license` id and `flickr_url`; the
  `licenses` array in the same file maps that id to the license name and its
  URI.
* In `data/face_detection.json`, the stem of every `file_name` is the Google
  Open Images ImageID of the source image, whose attribution and license
  metadata are published by Open Images.
* In `data/kitti_detection.json`, KITTI licenses its images as one dataset, so
  there are no per-image terms.

Apple makes no representations or warranties regarding the license status of
each source image, and you should verify the license for each image yourself.

## Acknowledgements

We gratefully acknowledge the creators of the source datasets:

* **COCO** — Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
* **KITTI** — Geiger et al., *Are we ready for Autonomous Driving? The KITTI
  Vision Benchmark Suite*, CVPR 2012.
* **Public Face Detection Dataset** — F. Elmenshawii, Kaggle face-detection
  dataset.

## Citation

If you use GESD or the GH-ESD method in your research, please cite:

```bibtex
@inproceedings{zhang2026ghesd,
  title={GH-ESD: Grounded Hypothesis-Driven Error Slice Discovery for Instance-Level Vision Tasks},
  author={Wei Zhang and Chaoqun Wang and Zixuan Guan and Ping Sheng Kao and Pengfei Zhao and Peng Wu and Sifeng He},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026},
  eprint={2512.24592},
  archivePrefix={arXiv}
}
```
