import json
from pathlib import Path

from detectron2.data import MetadataCatalog


MAPILLARY_DATASETS = (
    "mapillary_vistas_panoptic_train",
    "mapillary_vistas_panoptic_val",
    "mapillary_vistas_sem_seg_val",
)


def _mapillary_classes():
    label_path = (
        Path(__file__).resolve().parents[1]
        / "demo"
        / "label_files"
        / "mapillary-vistas-id2label.json"
    )
    with label_path.open("r") as f:
        id2label = json.load(f)
    return [id2label[str(i)] for i in range(len(id2label))]


def register_inference_metadata():
    classes = _mapillary_classes()
    metadata = {
        "stuff_classes": classes,
        "thing_classes": [],
        "stuff_dataset_id_to_contiguous_id": {i: i for i in range(len(classes))},
        "thing_dataset_id_to_contiguous_id": {},
        "evaluator_type": "sem_seg",
    }
    for name in MAPILLARY_DATASETS:
        MetadataCatalog.get(name).set(**metadata)
