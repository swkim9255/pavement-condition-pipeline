# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import os
import torch

from detectron2.data import MetadataCatalog
from defaults import DefaultPredictor
from visualizer import ColorMode, Visualizer


def _load_id2label(path):
    with open(path, "r") as f:
        id2label = json.load(f)
    return [id2label[str(i)] for i in range(len(id2label))]


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        if 'mapillary_vistas' in cfg.DATASETS.TEST_PANOPTIC[0]:
            label_path = os.path.join(
                os.path.dirname(__file__), "label_files", "mapillary-vistas-id2label.json"
            )
            classes = _load_id2label(label_path)
            self.metadata = self.metadata.set(
                stuff_classes=classes,
                thing_classes=[],
            )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = False
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, task):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        vis_output = {}
        
        if task == 'panoptic':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)
            predictions = self.predictor(image, task)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output['panoptic_inference'] = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(self.cpu_device), segments_info, alpha=0.7
        )

        if task == 'panoptic' or task == 'semantic':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
            predictions = self.predictor(image, task)
            vis_output['semantic_inference'] = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(self.cpu_device), alpha=0.7
            )

        if task == 'panoptic' or task == 'instance':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
            predictions = self.predictor(image, task)
            instances = predictions["instances"].to(self.cpu_device)
            vis_output['instance_inference'] = visualizer.draw_instance_predictions(predictions=instances, alpha=1)

        return predictions, vis_output
