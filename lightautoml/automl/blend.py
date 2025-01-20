"""Blenders."""

import logging

from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import cast
from typing import Dict

import numpy as np

from scipy.optimize import minimize_scalar

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.roles import NumericRole
from ..pipelines.ml.base import MLPipeline


logger = logging.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")


class Blender:
    """Basic class for blending.

    Blender learns how to make blend
    on sequence of prediction datasets and prune pipes,
    that are not used in final blend.

    """

    _outp_dim = None

    _bypass = False

    @property
    def outp_dim(self) -> int:  # noqa: D102
        return self._outp_dim

    def fit_predict(
        self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline], class_mapping: Dict
    ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        """Wraps custom ``._fit_predict`` methods of blenders.

        Method wraps individual ``._fit_predict`` method of blenders.
        If input is single model - take it, else ``._fit_predict``
        Note - some pipelines may have more than 1 model.
        So corresponding prediction dataset have multiple prediction cols.

        Args:
            predictions: Sequence of datasets with predictions.
            pipes: Sequence of pipelines.
            class_mapping: Mapping for target classes.

        Returns:
            Single prediction dataset and sequence of pruned pipelines.

        """
        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            self._bypass = True
            return predictions[0], pipes

        return self._fit_predict(predictions, pipes, class_mapping)

    def _fit_predict(
        self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline], class_mapping: Dict
    ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        """Defines how to fit, predict and prune - Abstract.

        Args:
            predictions: Sequence of datasets with predictions.
            pipes: Sequence of pipelines.
            class_mapping: Mapping for target classes.

        Returns:  # noqa: DAR202
            Single prediction dataset and sequence of pruned ``MLPipelines``.

        """
        raise NotImplementedError

    def predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Wraps custom ``._fit_predict`` methods of blenders.

        Args:
            predictions: Sequence of predictions from pruned datasets.

        Returns:
            Dataset with predictions.

        """
        if self._bypass:
            return predictions[0]

        return self._predict(predictions)

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Blend predictions on new sample.

        Args:
            predictions: Sequence of predictions from pruned datasets.

        Returns:  # noqa: DAR201
            Dataset with predictions.

        """
        raise NotImplementedError

    def split_models(self, predictions: Sequence[LAMLDataset]) -> Tuple[Sequence[LAMLDataset], List[int], List[int]]:
        """Split predictions by single model prediction datasets.

        Args:
            predictions: Sequence of datasets with predictions.

        Returns:
            Split predictions, model indices, pipe indices.

        """
        splitted_preds = []
        model_idx = []
        pipe_idx = []

        for n, preds in enumerate(predictions):

            features = preds.features
            n_models = len(features) // self.outp_dim

            for k in range(n_models):
                curr_pred = preds[:, features[k * self.outp_dim : (k + 1) * self.outp_dim]]
                splitted_preds.append(curr_pred)
                model_idx.append(k)
                pipe_idx.append(n)

        return splitted_preds, model_idx, pipe_idx

    def _set_metadata(self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline], class_mapping: Dict):

        pred0 = predictions[0]
        pipe0 = pipes[0]

        self._outp_dim = pred0.shape[1] // len(pipe0.ml_algos)
        self._outp_prob = pred0.task.name in ["binary", "multiclass"]
        self._score = predictions[0].task.get_dataset_metric()

        self._class_mapping = class_mapping

    def score(self, dataset: LAMLDataset) -> float:
        """Score metric for blender.

        Args:
            dataset: Blended predictions dataset.

        Returns:
            Metric value.

        """
        return self._score(dataset, True)


class BestModelSelector(Blender):
    """Select best single model from level.

    Drops pipes that are not used in calc best model.
    Works in general case (even on some custom things)
    and most efficient on inference.
    Perform worse than other on tables,
    specially if some of models was terminated by timer.

    """

    def _fit_predict(
        self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline], class_mapping: Dict
    ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        """Simple fit - just take one best.

        Args:
            predictions: Sequence of datasets with predictions.
            pipes: Sequence of pipelines.

        Returns:
            Single prediction dataset and Sequence of pruned pipelines.

        """
        self._set_metadata(predictions, pipes, class_mapping)
        splitted_preds, model_idx, pipe_idx = self.split_models(predictions)

        best_pred = None
        best_pipe_idx = 0
        best_model_idx = 0
        best_score = -np.inf

        for pred, mod, pipe in zip(splitted_preds, model_idx, pipe_idx):

            score = self.score(pred)

            if score > best_score:
                best_pipe_idx = pipe
                best_model_idx = mod
                best_score = score
                best_pred = pred

        best_pipe = pipes[best_pipe_idx]
        best_pipe.ml_algos = [best_pipe.ml_algos[best_model_idx]]

        return best_pred, [best_pipe]

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Simple predict - pruned pipe is a single model.

        Args:
            predictions: Sequence of predictions from pruned dataset.

        Returns:
            Dataset with predictions.

        """
        return predictions[0]


class MeanBlender(Blender):
    """Simple average level predictions.

    Works only with TabularDatasets.
    Doesn't require target to fit.
    No pruning.

    """

    def _get_mean_pred(self, splitted_preds: Sequence[NumpyDataset]) -> NumpyDataset:
        outp = splitted_preds[0].empty()

        pred = np.nanmean([x.data for x in splitted_preds], axis=0)

        outp.set_data(
            pred,
            ["MeanBlend_{0}".format(x) for x in range(pred.shape[1])],
            NumericRole(np.float32, prob=self._outp_prob),
        )

        return outp

    def _fit_predict(
        self, predictions: Sequence[NumpyDataset], pipes: Sequence[MLPipeline], class_mapping: Dict
    ) -> Tuple[NumpyDataset, Sequence[MLPipeline]]:
        """Simple fit_predict - just average and no prune.

        Args:
            predictions: Sequence of predictions.
            pipes: Sequence of pipelines.
            class_mapping: Mapping for target classes.

        Returns:
            Single prediction dataset and Sequence of pruned pipelines.

        """
        self._set_metadata(predictions, pipes, class_mapping)
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))

        outp = self._get_mean_pred(splitted_preds)

        return outp, pipes

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Simple fit_predict - just average.

        Args:
            predictions: Dataset with predictions.

        Returns:
            Dataset with averaged predictions.

        """
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))
        outp = self._get_mean_pred(splitted_preds)

        return outp


class WeightedBlender(Blender):
    """Weighted Blender based on coord descent, optimize task metric directly.

    Weight sum eq. 1.
    Good blender for tabular data,
    even if some predictions are NaN (ex. timeout).
    Model with low weights will be pruned.

    Args:
        max_iters: Max number of coord desc loops.
        max_inner_iters: Max number of iters to solve
            inner scalar optimization task.
        max_nonzero_coef: Maximum model weight value to stay in ensemble.

    """

    def __init__(
        self,
        max_iters: int = 5,
        max_inner_iters: int = 7,
        max_nonzero_coef: float = 0.05,
    ):
        self.max_iters = max_iters
        self.max_inner_iters = max_inner_iters
        self.max_nonzero_coef = max_nonzero_coef
        self.wts = [1]

    def _get_weighted_pred(self, splitted_preds: Sequence[NumpyDataset], wts: Optional[np.ndarray]) -> NumpyDataset:
        length = len(splitted_preds)
        if wts is None:
            wts = np.ones(length, dtype=np.float32) / length

        weighted_pred = np.nansum([x.data * w for (x, w) in zip(splitted_preds, wts)], axis=0).astype(np.float32)

        not_nulls = np.sum(
            [np.logical_not(np.isnan(x.data).any(axis=1)) * w for (x, w) in zip(splitted_preds, wts)],
            axis=0,
        ).astype(np.float32)

        not_nulls = not_nulls[:, np.newaxis]

        weighted_pred /= not_nulls
        weighted_pred = np.where(not_nulls == 0, np.nan, weighted_pred)

        outp = splitted_preds[0].empty()
        outp.set_data(
            weighted_pred,
            self._class_mapping if self._class_mapping else list(range(weighted_pred.shape[1])),
            NumericRole(np.float32, prob=self._outp_prob),
        )

        return outp

    def _get_candidate(self, wts: np.ndarray, idx: int, value: float) -> np.ndarray:
        """Replaces the value at the idx position in the input array with a new input value, \
        modifies the values in all other positions to keep the total sum of the array equal to 1.

        Args:
            wts (np.ndarray): input array
            idx (int): index
            value (float): new value

        Returns:
            np.ndarray: new array with sum equal to 1.
        """
        candidate = wts.copy()
        sl = np.arange(wts.shape[0]) != idx
        s = candidate[sl].sum()
        candidate[sl] = candidate[sl] / s * (1 - value)
        candidate[idx] = value

        # this is the part for pipeline pruning
        order = candidate.argsort()
        for idx in order:
            if candidate[idx] < self.max_nonzero_coef:
                candidate[idx] = 0
                candidate /= candidate.sum()
            else:
                break

        return candidate

    def _get_scorer(self, splitted_preds: Sequence[NumpyDataset], idx: int, wts: np.ndarray) -> Callable:
        def scorer(x):
            candidate = self._get_candidate(wts, idx, x)

            pred = self._get_weighted_pred(splitted_preds, candidate)
            score = self.score(pred)

            return -score

        return scorer

    def _optimize(self, splitted_preds: Sequence[NumpyDataset], eps: float = 1e-7) -> np.ndarray:
        """Perform coordinate descent.

        Args:
            splitted_preds (Sequence[NumpyDataset]): predictions for weighting to maximize the metric.
            eps (float): epsilon.

        Returns:
            np.ndarray: best weights.
        """
        # the set of initial weights for blending
        candidate = np.ones(len(splitted_preds), dtype=np.float32) / len(splitted_preds)

        best_score = self.score(self._get_weighted_pred(splitted_preds, candidate))
        best_weights = candidate
        logger.info(f"Blending: optimization starts with equal weights. Score = \x1b[1m{best_score:.7f}\x1b[0m")

        for iteration_num in range(self.max_iters):
            flg_no_upd = True
            for weights_idx in range(len(splitted_preds)):
                if candidate[weights_idx] == 1:
                    continue

                opt_res = minimize_scalar(
                    self._get_scorer(splitted_preds, weights_idx, candidate),
                    method="Bounded",
                    bounds=(0, 1),
                    options={"disp": False, "maxiter": self.max_inner_iters},
                )
                new_weight = opt_res.x
                score = -opt_res.fun
                candidate = self._get_candidate(candidate, weights_idx, new_weight)

                if score - eps > best_score:
                    flg_no_upd = False
                    best_score = score
                    best_weights = candidate

            if flg_no_upd:
                logger.info("Blending: no improvements for score. Terminated.\n")
                break
            else:
                logger.info(
                    f"Blending: iteration \x1b[1m{iteration_num}\x1b[0m: score = \x1b[1m{best_score:.7f}\x1b[0m, weights = \x1b[1m{best_weights}\x1b[0m"
                )

        logger.info(
            f"Blending: best score = \x1b[1m{best_score:.7f}\x1b[0m, best weights = \x1b[1m{best_weights}\x1b[0m"
        )

        return best_weights

    @staticmethod
    def _prune_pipe(
        pipes: Sequence[MLPipeline], wts: np.ndarray, pipe_idx: np.ndarray
    ) -> Tuple[Sequence[MLPipeline], np.ndarray]:
        new_pipes = []

        for i in range(max(pipe_idx) + 1):
            pipe = pipes[i]
            weights = wts[np.array(pipe_idx) == i]

            pipe.ml_algos = [x for (x, w) in zip(pipe.ml_algos, weights) if w > 0]

            new_pipes.append(pipe)

        new_pipes = [x for x in new_pipes if len(x.ml_algos) > 0]
        wts = wts[wts > 0]
        return new_pipes, wts

    def _fit_predict(
        self, predictions: Sequence[NumpyDataset], pipes: Sequence[MLPipeline], class_mapping: Dict
    ) -> Tuple[NumpyDataset, Sequence[MLPipeline]]:
        """Perform coordinate descent.

        Args:
            predictions: Sequence of prediction datasets.
            pipes: Sequence of pipelines.
            class_mapping: Mapping for target classes.

        Returns:
            Single prediction dataset and Sequence of pruned pipelines.

        Returns:
            Dataset and MLPipeline.

        """
        self._set_metadata(predictions, pipes, class_mapping)
        splitted_preds, _, pipe_idx = cast(List[NumpyDataset], self.split_models(predictions))

        wts = self._optimize(splitted_preds)
        splitted_preds = [x for (x, w) in zip(splitted_preds, wts) if w > 0]
        pipes, self.wts = self._prune_pipe(pipes, wts, pipe_idx)

        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp, pipes

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Simple - weighted average.

        Args:
            predictions: Sequence of predictions.

        Returns:
            Dataset with weighted predictions.

        """
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))
        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp
