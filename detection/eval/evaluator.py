import numpy as np

from . import detection


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        raise NotImplementedError ("[reset] method need to be implemented in child class.")

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        raise NotImplementedError ("[process] method need to be implemented in child class.")

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        """
        raise NotImplementedError ("[evaluate] method need to be implemented in child class.")


class DetectionEvaluator(DatasetEvaluator):
    """
    Evaluator for detection task.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, iou_thresh=0.5):
        self._evaluator = detection.Evaluator()
        self._iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        self._bbox = detection.BoundingBoxes()

    def process(self, groudtruths, predictions):
        """
        Inputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-input-format
        Outputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-output-format
        """
        for sample_input, sample_output in zip(groudtruths, predictions):
            image_id = sample_input['image_id']
            gt_instances = sample_input['instances']
            pred_instances = sample_output['instances']
            width = sample_input['width']
            height = sample_input['height']
            for i in range(len(gt_instances)):
                instance = gt_instances[i]
                class_id = instance.get(
                    'gt_classes').cpu().detach().numpy().item()
                boxes = instance.get('gt_boxes')
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.GroundTruth,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)
            for i in range(len(pred_instances)):
                instance = pred_instances[i]
                class_id = instance.get(
                    'pred_classes').cpu().detach().numpy().item()
                scores = instance.get('scores').cpu().detach().numpy().item()
                boxes = instance.get('pred_boxes')
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.Detected,
                        scores,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)

    def evaluate(self):
        results = self._evaluator.GetPascalVOCMetrics(self._bbox, self._iou_thresh)
        if isinstance(results, dict):
            results = [results]
        metrics = {}
        APs = []
        for result in results:
            metrics[f'AP_{result["class"]}'] = result['AP']
            APs.append(result['AP'])
        metrics['mAP'] = np.nanmean(APs)
        self._evaluator.PlotPrecisionRecallCurve(self._bbox, savePath="./plots/", showGraphic=False)
        return metrics

class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
