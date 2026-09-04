# GESD Error-Slice Mapping Reference

This document records how the raw annotation `reason` codes in the original dumps
were normalized into the **canonical ground-truth error-slice names** used in the
public release. Canonical names are taken verbatim from the paper appendix tables
(`tab:fesd_detection_detailed` and `tab:fesd_segmentation_results`).

After cleaning, every annotation has:

- `reason` — the canonical slice name, or `"NA"` for a correct (non-error) instance;
- `reason_code` — the original raw code for error instances; `"NA"` for correct (non-error) instances;
- `result` — `TP` / `FP` / `FN` (detection files only).

**Correct-instance unification.** Correct detections previously carried either a
category name (`person`, `face`, …) or `None`/`null` in the `reason` field. All such
instances (`result == "TP"`, and the clean ground-truth instances in the
segmentation file) are unified to `reason == "NA"`.

## Object Detection — 21 canonical slices

The detection benchmark pools three image sources (COCO, KITTI, face); a slice such
as *Blurry face missed* legitimately appears in more than one file.

| Raw code(s) | Canonical GT slice name |
|---|---|
| `stroller_trolley` | Stroller trolley is mistakenly detected as a bicycle |
| `blurry_face` | Blurry face missed |
| `partial80car` | Cars that are partially obscured missed. |
| `nonphysical_face` | Nonphysical face(e.g., face reflected in a mirror) is mistakenly detected as face |
| `paper` | Paper is mistakenly detected as book |
| `stop_sign_fail` | Red traffic sign is mistakenly detected as stop sign |
| `person_in_car` | Person in the car missed |
| `close` | Closed umbrella missed |
| `artistic_face` | Artistic face is mistakenly detected as face |
| `partial_face` | Partial face missed |
| `no_flower` | Vase with no flower or plants inside missed |
| `baby_face` | Baby face missed |
| `trailer` | Trailer is mistakenly detected as truck |
| `dumpster` | Dumpster is mistakenly detected as truck |
| `artificial_face` | Artificial face is mistakenly detected as face |
| `open_book` | Open book missed |
| `watch` | Watch missed when detecting clock |
| `open_suitcase` | Open suitcase missed |
| `cycle_wrider` | Bicycles seen from the front/back that being ridden/pushed by people |
| `car_open_door` | Cars with open doors are mistakenly detected as truck |
| `nonvertical_face` | Nonvertical face in the image missed |

## Instance Segmentation — 21 canonical slices

The raw codes `slice` and `crowd` are **split by the ground-truth category** into
the distinct per-fruit slices the paper enumerates separately.

| Raw code (+ gt category) | Canonical GT slice name |
|---|---|
| `crowd` + banana | Clustered bananas merged, boundaries unclear |
| `occlusion` | Airplane boundary expanded under occlusion |
| `pet_on` | Bed over-segmented with pet |
| `laptop_keyboard` | Laptop keyboard merged with laptop body |
| `paper` | Paper mis-segmented as book |
| `apple_logo` | Apple logo mis-segmented as apple |
| `similar_cloth_color` | Tie boundary unclear due to color blending |
| `slice` + carrot | Sliced carrot under-segmented due to shape change |
| `toy_near_by` | Adjacent toys merged with teddy bear |
| `crowd` + orange | Clustered oranges merged, boundaries unclear |
| `open_suitcase` | Open suitcase missed (non-canonical form) |
| `open_book` | Open book missed due to shape variation |
| `crowd` + apple | Clustered apples merged, boundaries unclear |
| `slice` + banana | Sliced banana under-segmented due to shape change |
| `close` | Closed umbrella missed (non-typical pose) |
| `no_flower` | Vase without flowers missed due to context prior |
| `slice` + orange | Sliced orange under-segmented due to shape change |
| `person_on` | Bed over-segmented together with person |
| `onland` | Surfboard on land missed (expected water context) |
| `watch` | Watch missed when segmenting clock |
| `slice` + apple | Sliced apple under-segmented due to shape change |

All 21 segmentation slice counts in the cleaned release match the GT sizes reported
in the paper appendix exactly.
