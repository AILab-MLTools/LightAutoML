# TODO: проверить что NLP не падает, CV
"""Tabular presets."""

import logging
import os

from collections import Counter
from copy import copy
from copy import deepcopy
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from joblib import Parallel
from joblib import delayed
from pandas import DataFrame
from tqdm import tqdm

from ...addons.utilization import TimeUtilization
from ...dataset.np_pd_dataset import NumpyDataset
from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.boost_lgbm import BoostLGBM
from ...ml_algo.boost_xgb import BoostXGB
from ...ml_algo.dl_model import TorchModel
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.random_forest import RandomForestSklearn
from ...ml_algo.tuning.optuna import OptunaTuner
from ...pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from ...pipelines.features.lgb_pipeline import LGBSeqSimpleFeatures
from ...pipelines.features.lgb_pipeline import LGBSimpleFeatures
from ...pipelines.features.linear_pipeline import LinearFeatures
from ...pipelines.features.torch_pipeline import TorchSimpleFeatures
from ...pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from ...pipelines.selection.base import ComposedSelector
from ...pipelines.selection.base import SelectionPipeline
from ...pipelines.selection.importance_based import ImportanceCutoffSelector
from ...pipelines.selection.importance_based import ModelBasedImportanceEstimator
from ...pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from ...pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from ...reader.base import DictToPandasSeqReader
from ...reader.base import PandasToPandasReader
from ...reader.tabular_batch_generator import ReadableToDf
from ...reader.tabular_batch_generator import read_batch
from ...reader.tabular_batch_generator import read_data
from ...tasks import Task
from ..blend import MeanBlender
from ..blend import WeightedBlender
from .base import AutoMLPreset
from .base import upd_params
from .utils import calc_feats_permutation_imps
from .utils import change_datetime
from .utils import plot_pdp_with_distribution


_base_dir = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


class TabularAutoML(AutoMLPreset):
    """Classic preset - work with tabular data.

    Supported data roles - numbers, dates, categories.
    Limitations:

        - No memory management
        - No text support

    GPU support in catboost/lightgbm (if installed for GPU) training.

    Commonly _params kwargs (ex. timing_params) set via
    config file (config_path argument).
    If you need to change just few params, it's possible
    to pass it as dict of dicts, like json.
    To get available params please look on default config template.
    Also you can find there param description.
    To generate config template call
    :meth:`TabularAutoML.get_config('config_path.yml')`.

    Args:
        task: Task to solve.
        timeout: Timeout in seconds.
        memory_limit: Memory limit that are passed to each automl.
        cpu_limit: CPU limit that that are passed to each automl.
        gpu_ids: GPU IDs that are passed to each automl.
        debug: To catch running model exceptions or not.
        timing_params: Timing param dict. Optional.
        config_path: Path to config file.
        general_params: General param dict.
        reader_params: Reader param dict.
        read_csv_params: Params to pass ``pandas.read_csv``
            (case of train/predict from file).
        nested_cv_params: Param dict for nested cross-validation.
        tuning_params: Params of Optuna tuner.
        selection_params: Params of feature selection.
        lgb_params: Params of lightgbm model.
        cb_params: Params of catboost model.
        xgb_params: Params of xgboost model.
        rf_params: Params of Sklearn Random Forest model.
        linear_l2_params: Params of linear model.
        nn_params: Params of neural network model.
        gbm_pipeline_params: Params of feature generation
            for boosting models.
        linear_pipeline_params: Params of feature generation
            for linear models.
        nn_pipeline_params: Params of feature generation
            for neural network models.
    """

    _default_config_path = "tabular_config.yml"

    # set initial runtime rate guess for first level models
    _time_scores = {
        "lgb": 1,
        "lgb_tuned": 3,
        "xgb": 1,
        "xgb_tuned": 3,
        "linear_l2": 0.7,
        "cb": 2,
        "cb_tuned": 6,
        "rf": 5,
        "rf_tuned": 10,
        "nn": 10,
        "nn_tuned": 20,
    }

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = "all",
        debug: bool = False,
        timing_params: Optional[dict] = None,
        config_path: Optional[str] = None,
        general_params: Optional[dict] = None,
        reader_params: Optional[dict] = None,
        read_csv_params: Optional[dict] = None,
        nested_cv_params: Optional[dict] = None,
        tuning_params: Optional[dict] = None,
        selection_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        rf_params: Optional[dict] = None,
        linear_l2_params: Optional[dict] = None,
        nn_params: Optional[dict] = None,
        gbm_pipeline_params: Optional[dict] = None,
        linear_pipeline_params: Optional[dict] = None,
        nn_pipeline_params: Optional[dict] = None,
        time_series_pipeline_params: Optional[dict] = None,
        is_time_series: bool = False,
    ):
        super().__init__(task, timeout, memory_limit, cpu_limit, gpu_ids, debug, timing_params, config_path)
        #
        self.is_time_series = is_time_series

        # upd manual params
        for name, param in zip(
            [
                "general_params",
                "reader_params",
                "read_csv_params",
                "nested_cv_params",
                "tuning_params",
                "lgb_params",
                "cb_params",
                "xgb_params",
                "rf_params",
                "linear_l2_params",
                "nn_params",
                "gbm_pipeline_params",
                "linear_pipeline_params",
                "nn_pipeline_params",
            ],
            [
                general_params,
                reader_params,
                read_csv_params,
                nested_cv_params,
                tuning_params,
                lgb_params,
                cb_params,
                xgb_params,
                rf_params,
                linear_l2_params,
                nn_params,
                gbm_pipeline_params,
                linear_pipeline_params,
                nn_pipeline_params,
            ],
        ):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)

        # if not time-series mode --> update selection_params too
        if not self.is_time_series:
            for name, param in zip(["selection_params"], [selection_params]):
                if param is None:
                    param = {}
                self.__dict__[name] = upd_params(self.__dict__[name], param)

        # if time-series mode --> update time_series_pipeline_params
        if self.is_time_series:
            for name, param in zip(["time_series_pipeline_params"], [time_series_pipeline_params]):
                if param is None:
                    param = {}
                self.__dict__[name] = upd_params(self.__dict__[name], param)

    def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):

        length = train_data.shape[0]

        # infer optuna tuning iteration based on dataframe len
        if self.tuning_params["max_tuning_iter"] == "auto":
            if length < 10000:
                self.tuning_params["max_tuning_iter"] = 100
            elif length < 30000:
                self.tuning_params["max_tuning_iter"] = 50
            elif length < 100000:
                self.tuning_params["max_tuning_iter"] = 10
            else:
                self.tuning_params["max_tuning_iter"] = 5

        if self.general_params["use_algos"] == "auto":
            # TODO: More rules and add cases
            self.general_params["use_algos"] = [["lgb", "lgb_tuned", "linear_l2", "cb", "cb_tuned"]]
            if self.task.name == "multi:reg" and self.is_time_series:
                self.general_params["use_algos"] = [["cb", "linear_l2", "rf"]]
            else:
                if self.task.name == "multiclass" and multilevel_avail:
                    self.general_params["use_algos"].append(["linear_l2", "lgb"])

                if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                    self.general_params["use_algos"] = [["linear_l2", "cb", "rf", "rf_tuned", "cb_tuned"]]

        if not self.general_params["nested_cv"]:
            self.nested_cv_params["cv"] = 1

        # check gpu to use catboost
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.nn_params["device"] = gpu_ids.split(",")
            self.cb_params["default_params"]["task_type"] = "GPU"
            self.cb_params["default_params"]["devices"] = gpu_ids.replace(",", ":")

            self.xgb_params["default_params"]["device"] = f"cuda:{gpu_ids.split(',')[-1]}"  # TODO: add multigpu for xgb
        else:
            self.nn_params["device"] = "cpu"

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.cb_params["default_params"]["thread_count"] = min(
            self.cb_params["default_params"]["thread_count"], cpu_cnt
        )
        self.lgb_params["default_params"]["num_threads"] = min(
            self.lgb_params["default_params"]["num_threads"], cpu_cnt
        )
        self.xgb_params["default_params"]["nthread"] = min(self.xgb_params["default_params"]["nthread"], cpu_cnt)
        self.reader_params["n_jobs"] = min(self.reader_params["n_jobs"], cpu_cnt)

    def get_feature_pipeline(self, model, **kwargs):
        """Get LGBSeqSimpleFeatures pipeline if task is the time series prediction.

        Args:
            model: one from ["gbm", "linear_l2",, "rf", "nn"].
            kwargs: Arbitrary keyword arguments.

        Returns:
            appropriate features pipeline.
        """
        if self.is_time_series and model in ["gbm", "linear_l2", "rf", "nn"]:
            return LGBSeqSimpleFeatures(fill_na=True, scaler=True, transformers_params=self.time_series_pipeline_params)
        else:
            if model == "nn":
                return TorchSimpleFeatures(**self.nn_pipeline_params)
            if model == "linear_l2":
                return LinearFeatures(output_categories=True, **self.linear_pipeline_params)
            if model == "gbm":
                return LGBAdvancedPipeline(**self.gbm_pipeline_params, **kwargs)
            if model == "rf":
                if "fill_na" in kwargs:
                    return LGBAdvancedPipeline(**self.gbm_pipeline_params, **kwargs)
                return LGBAdvancedPipeline(**self.gbm_pipeline_params, fill_na=True, **kwargs)

    def get_time_score(self, n_level: int, model_type: str, nested: Optional[bool] = None):

        if nested is None:
            nested = self.general_params["nested_cv"]

        score = self._time_scores[model_type]

        mult = 1
        if nested:
            if self.nested_cv_params["n_folds"] is not None:
                mult = self.nested_cv_params["n_folds"]
            else:
                mult = self.nested_cv_params["cv"]

        if n_level > 1:
            mult *= 0.8 if self.general_params["skip_conn"] else 0.1

        score = score * mult

        # lower score for catboost on gpu
        if model_type in ["cb", "cb_tuned"] and self.cb_params["default_params"]["task_type"] == "GPU":
            score *= 0.5
        return score

    def get_selector(self, n_level: Optional[int] = 1) -> SelectionPipeline:
        selection_params = self.selection_params
        # lgb_params
        lgb_params = deepcopy(self.lgb_params)
        lgb_params["default_params"] = {
            **lgb_params["default_params"],
            **{"feature_fraction": 1},
        }

        cb_params = deepcopy(self.cb_params)
        cb_params["default_params"] = {
            **cb_params["default_params"],
            **{"rsm": 1},
        }

        mode = selection_params["mode"]

        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            # timer will be useful to estimate time for next gbm runs
            selection_feats = LGBSimpleFeatures()

            if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                time_score = self.get_time_score(n_level, "cb", False)
                sel_timer_0 = self.timer.get_task_timer("cb", time_score)
                selection_gbm = BoostCB(timer=sel_timer_0, **cb_params)
                model_name = "cb"
            else:
                time_score = self.get_time_score(n_level, "lgb", False)
                sel_timer_0 = self.timer.get_task_timer("lgb", time_score)
                selection_gbm = BoostLGBM(timer=sel_timer_0, **lgb_params)
                model_name = "lgb"
            selection_gbm.set_prefix("Selector")
            time_score = self.get_time_score(n_level, model_name, False)

            sel_timer_0 = self.timer.get_task_timer(model_name, time_score)

            if selection_params["importance_type"] == "permutation":
                importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(
                selection_feats,
                selection_gbm,
                importance,
                cutoff=selection_params["cutoff"],
                fit_on_holdout=selection_params["fit_on_holdout"],
            )
            if mode == 2:
                time_score = self.get_time_score(n_level, model_name, False)

                sel_timer_1 = self.timer.get_task_timer(model_name, time_score)
                selection_feats = LGBSimpleFeatures()
                if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                    selection_gbm = BoostCB(timer=sel_timer_1, **cb_params)
                else:
                    selection_gbm = BoostLGBM(timer=sel_timer_1, **lgb_params)
                selection_gbm.set_prefix("Selector")

                # TODO: Check about reusing permutation importance
                importance = NpPermutationImportanceEstimator()

                extra_selector = NpIterativeFeatureSelector(
                    selection_feats,
                    selection_gbm,
                    importance,
                    feature_group_size=selection_params["feature_group_size"],
                    max_features_cnt_in_result=selection_params["max_features_cnt_in_result"],
                )

                pre_selector = ComposedSelector([pre_selector, extra_selector])

        return pre_selector

    def get_nn(
        self, keys: Sequence[str], n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None
    ) -> NestedTabularMLPipeline:
        ml_algos = []
        force_calc = []

        nn_feats = self.get_feature_pipeline(model="nn")
        general_nn_params = deepcopy(self.nn_params)
        if "0" in self.nn_params:
            for i in range(len(keys)):
                if str(i) in general_nn_params:
                    del general_nn_params[str(i)]

        for i, key in enumerate(keys):
            time_score = self.get_time_score(n_level, "nn")
            nn_timer = self.timer.get_task_timer("reg_nn", time_score)
            model_params = deepcopy(general_nn_params)

            model_params.update(self.nn_params.get(str(i), general_nn_params))
            model_name = key

            if isinstance(key, str):
                tuned = "_tuned" in key
                if key[:2] == "nn":
                    model_name = "mlp"
                _name = "TorchNN_" + model_name + "_" + str(i)
                model_name = model_name.replace("_tuned", "")
            else:
                tuned = model_params.get("tuned")
                _name = "TorchNN_" + str(i)

            model_params["model"] = model_name
            nn_model = TorchModel(
                timer=nn_timer,
                default_params=model_params,
                freeze_defaults=model_params["freeze_defaults"],
                optimization_search_space=model_params.get("optimization_search_space", None),
            )
            nn_model._name = _name

            if tuned:
                nn_model.set_prefix("Tuned")
                nn_tuner = OptunaTuner(
                    n_trials=model_params["tuning_params"]["max_tuning_iter"],
                    timeout=model_params["tuning_params"]["max_tuning_time"],
                    fit_on_holdout=model_params["tuning_params"]["fit_on_holdout"],
                )
                nn_model = (nn_model, nn_tuner)
            ml_algos.append(nn_model)
            force_calc.append(True if not len(ml_algos) - 1 else False)

        nn_pipe = NestedTabularMLPipeline(
            ml_algos, force_calc, pre_selection=None, features_pipeline=nn_feats, **self.nested_cv_params
        )

        return nn_pipe

    def get_linear(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)
        linear_l2_feats = self.get_feature_pipeline(model="linear_l2")

        linear_l2_pipe = NestedTabularMLPipeline(
            [linear_l2_model],
            force_calc=True,
            pre_selection=pre_selector,
            features_pipeline=linear_l2_feats,
            **self.nested_cv_params,
        )
        return linear_l2_pipe

    def get_gbms(
        self,
        keys: Sequence[str],
        n_level: int = 1,
        pre_selector: Optional[SelectionPipeline] = None,
    ):

        gbm_feats = self.get_feature_pipeline(model="gbm", feats_imp=pre_selector)

        ml_algos = []
        force_calc = []
        for idx, key in enumerate(keys):
            if idx == 0:  # force calculations only for the first ml_algo
                force = True

            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            if algo_key == "lgb":
                gbm_model = BoostLGBM(timer=gbm_timer, **self.lgb_params)
            elif algo_key == "cb":
                gbm_model = BoostCB(timer=gbm_timer, **self.cb_params)
            elif algo_key == "xgb":
                gbm_model = BoostXGB(timer=gbm_timer, **self.xgb_params)
            else:
                raise ValueError("Wrong algo key")

            if tuned:
                gbm_model.set_prefix("Tuned")
                gbm_tuner = OptunaTuner(
                    n_trials=self.tuning_params["max_tuning_iter"],
                    timeout=self.tuning_params["max_tuning_time"],
                    fit_on_holdout=self.tuning_params["fit_on_holdout"],
                )
                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)
            force = False

        gbm_pipe = NestedTabularMLPipeline(
            ml_algos, force_calc, pre_selection=pre_selector, features_pipeline=gbm_feats, **self.nested_cv_params
        )

        return gbm_pipe

    def get_rfs(self, keys: Sequence[str], n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None):

        rf_feats = self.get_feature_pipeline(model="rf", feats_imp=pre_selector, fill_na=True)
        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False]):
            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            rf_timer = self.timer.get_task_timer(algo_key, time_score)

            rf_model = RandomForestSklearn(timer=rf_timer, **self.rf_params)

            if tuned:
                rf_model.set_prefix("Tuned")
                rf_tuner = OptunaTuner(
                    n_trials=self.tuning_params["max_tuning_iter"],
                    timeout=self.tuning_params["max_tuning_time"],
                    fit_on_holdout=self.tuning_params["fit_on_holdout"],
                )
                rf_model = (rf_model, rf_tuner)
            ml_algos.append(rf_model)
            force_calc.append(force)

        rf_pipe = NestedTabularMLPipeline(
            ml_algos, force_calc, pre_selection=pre_selector, features_pipeline=rf_feats, **self.nested_cv_params
        )

        return rf_pipe

    def create_automl(self, **fit_args):
        """Create basic automl instance.

        Args:
            **fit_args: Contain all information needed for creating automl.

        """
        train_data = fit_args["train_data"]
        multilevel_avail = fit_args["valid_data"] is None and fit_args["cv_iter"] is None

        if self.is_time_series:
            self.infer_auto_params(train_data["seq"]["seq0"], multilevel_avail)
            reader = DictToPandasSeqReader(task=self.task, **self.reader_params)
            pre_selector = None
        else:
            self.infer_auto_params(train_data, multilevel_avail)
            reader = PandasToPandasReader(task=self.task, **self.reader_params)
            pre_selector = self.get_selector()
        levels = []

        for n, names in enumerate(self.general_params["use_algos"]):
            lvl = []
            # regs
            rf_models = [x for x in ["rf", "rf_tuned"] if x in names]

            if len(rf_models) > 0:
                selector = None
                if (
                    self.is_time_series
                    or "rf" in self.selection_params["select_algos"]
                    and (self.general_params["skip_conn"] or n == 0)
                ):
                    selector = pre_selector
                lvl.append(self.get_rfs(rf_models, n + 1, selector))

            if "linear_l2" in names:
                selector = None
                if (
                    self.is_time_series
                    or "linear_l2" in self.selection_params["select_algos"]
                    and (self.general_params["skip_conn"] or n == 0)
                ):
                    selector = pre_selector
                lvl.append(self.get_linear(n + 1, selector))

            gbm_models = [
                x
                for x in ["lgb", "lgb_tuned", "cb", "cb_tuned", "xgb", "xgb_tuned"]
                if x in names and x.split("_")[0] in self.task.losses
            ]

            if len(gbm_models) > 0:
                selector = None
                if (
                    self.is_time_series
                    or "gbm" in self.selection_params["select_algos"]
                    and (self.general_params["skip_conn"] or n == 0)
                ):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            available_nn_models = [
                "nn",
                "mlp",
                "dense",
                "denselight",
                "resnet",
                "snn",
                "linear_layer",
                "_linear_layer",
                "node",
                "autoint",
                "tabnet",
                "fttransformer",
                "saint",
            ]
            available_nn_models = available_nn_models + [x + "_tuned" for x in available_nn_models]
            nn_models = [
                x for x in names if x in available_nn_models or (isinstance(x, type) and issubclass(x, nn.Module))
            ]

            if len(nn_models) > 0:
                selector = None
                lvl.append(self.get_nn(nn_models, n + 1, selector))

            if len(lvl) != 0:
                levels.append(lvl)

        # blend everything
        blender = WeightedBlender(max_nonzero_coef=self.general_params["weighted_blender_max_nonzero_coef"])

        # initialize
        self._initialize(
            reader,
            levels,
            skip_conn=self.general_params["skip_conn"],
            blender=blender,
            return_all_predictions=self.general_params["return_all_predictions"],
            timer=self.timer,
            debug=self.debug,
        )

    def _get_read_csv_params(self):
        try:
            cols_to_read = self.reader.used_features
            numeric_dtypes = {
                x: self.reader.roles[x].dtype for x in self.reader.roles if self.reader.roles[x].name == "Numeric"
            }
        except AttributeError:
            cols_to_read = []
            numeric_dtypes = {}
        # cols_to_read is empty if reader is not fitted
        if len(cols_to_read) == 0:
            cols_to_read = None

        read_csv_params = copy(self.read_csv_params)
        read_csv_params = {
            **read_csv_params,
            **{"usecols": cols_to_read, "dtype": numeric_dtypes},
        }

        return read_csv_params

    def fit_predict(
        self,
        train_data: ReadableToDf,
        roles: Optional[dict] = None,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[ReadableToDf] = None,
        valid_features: Optional[Sequence[str]] = None,
        log_file: str = None,
        verbose: int = 0,
        path_to_save: Optional[str] = None,
    ) -> NumpyDataset:
        """Fit and get prediction on validation dataset.

        Almost same as :meth:`lightautoml.automl.base.AutoML.fit_predict`.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names, if can't be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example, :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features if cannot be inferred from `valid_data`.
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
            log_file: Filename for writing logging messages. If log_file is specified,
                the messages will be saved in a the file. If the file exists, it will be overwritten.
            path_to_save: The path that joblib will use to save the model after fit stage is completed. Use *.joblib format.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        # roles may be none in case of train data is set {'data': np.ndarray, 'target': np.ndarray ...}
        self.set_logfile(log_file)
        if roles is None:
            roles = {}
        read_csv_params = self._get_read_csv_params()
        if self.is_time_series:
            train_data = train_data["seq"]["seq0"]
        train, upd_roles = read_data(train_data, train_features, self.cpu_limit, read_csv_params)
        if upd_roles:
            roles = {**roles, **upd_roles}
        if valid_data is not None:
            data, _ = read_data(valid_data, valid_features, self.cpu_limit, self.read_csv_params)

        if self.is_time_series:
            train = {"seq": {"seq0": train}}

        oof_pred = super().fit_predict(
            train,
            roles=roles,
            cv_iter=cv_iter,
            valid_data=valid_data,
            verbose=verbose,
            path_to_save=path_to_save,
        )

        return cast(NumpyDataset, oof_pred)

    def predict(
        self,
        data: ReadableToDf,
        features_names: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = 1,
        return_all_predictions: Optional[bool] = None,
    ) -> NumpyDataset:
        """Get dataset with predictions.

        Almost same as :meth:`lightautoml.automl.base.AutoML.predict`
        on new dataset, with additional features.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`. For example,
              ``{'data': X...}``. In this case roles are optional,
              but `train_features` and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Parallel inference - you can pass ``n_jobs`` to speedup
        prediction (requires more RAM).
        Batch_inference - you can pass ``batch_size``
        to decrease RAM usage (may be longer).

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
                if cannot be inferred from `train_data`.
            batch_size: Batch size or ``None``.
            n_jobs: Number of jobs.
            return_all_predictions: if True,
                returns all model predictions from last level

        Returns:
            Dataset with predictions.

        """
        read_csv_params = self._get_read_csv_params()

        if batch_size is None and n_jobs == 1:
            if self.is_time_series:
                data = data["seq"]["seq0"]
            data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
            if self.is_time_series:
                data = {"seq": {"seq0": data}}
            pred = super().predict(data, features_names, return_all_predictions)
            return cast(NumpyDataset, pred)

        data_generator = read_batch(
            data,
            features_names,
            n_jobs=n_jobs,
            batch_size=batch_size,
            read_csv_params=read_csv_params,
        )

        if n_jobs == 1:
            res = [self.predict(df, features_names, return_all_predictions) for df in data_generator]
        else:
            # TODO: Check here for pre_dispatch param
            with Parallel(n_jobs, pre_dispatch=len(data_generator) + 1) as p:
                res = p(delayed(self.predict)(df, features_names, return_all_predictions) for df in data_generator)

        res = NumpyDataset(
            np.concatenate([x.data for x in res], axis=0),
            features=res[0].features,
            roles=res[0].roles,
        )

        return res

    def get_feature_scores(
        self,
        calc_method: str = "fast",
        data: Optional[ReadableToDf] = None,
        features_names: Optional[Sequence[str]] = None,
        silent: bool = True,
    ):
        if calc_method == "fast":
            for level in self.levels:
                for pipe in level:
                    fi = pipe.pre_selection.get_features_score()
                    if fi is not None:
                        used_feats = set(self.collect_used_feats())
                        fi = fi.reset_index()
                        fi.columns = ["Feature", "Importance"]
                        return fi[fi["Feature"].map(lambda x: x in used_feats)]

            else:
                if not silent:
                    logger.info2("No feature importances to show. Please use another calculation method")
                return None

        if calc_method != "accurate":
            if not silent:
                logger.info2(
                    "Unknown calc_method. "
                    + "Currently supported methods for feature importances calculation are 'fast' and 'accurate'."
                )
            return None

        if data is None:
            if not silent:
                logger.info2("Data parameter is not setup for accurate calculation method. Aborting...")
            return None

        read_csv_params = self._get_read_csv_params()
        data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
        used_feats = self.collect_used_feats()
        fi = calc_feats_permutation_imps(
            self,
            used_feats,
            data,
            self.reader.target,
            self.task.get_dataset_metric(),
            silent=silent,
        )
        return fi

    def get_individual_pdp(
        self,
        test_data: ReadableToDf,
        feature_name: str,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
    ):
        assert feature_name in self.reader._roles
        assert datetime_level in ["year", "month", "dayofweek"]
        test_i = test_data.copy()
        # Numerical features
        if self.reader._roles[feature_name].name == "Numeric":
            counts, bin_edges = np.histogram(test_data[feature_name].dropna(), bins=n_bins)
            grid = (bin_edges[:-1] + bin_edges[1:]) / 2
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = i
                preds = self.predict(test_i).data
                ys.append(preds)
        # Categorical features
        if self.reader._roles[feature_name].name == "Category":
            feature_cnt = test_data[feature_name].value_counts()
            grid = list(feature_cnt.index.values[:top_n_categories])
            counts = list(feature_cnt.values[:top_n_categories])
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = i
                preds = self.predict(test_i).data
                ys.append(preds)
            if len(feature_cnt) > top_n_categories:
                freq_mapping = {feature_cnt.index[i]: i for i, _ in enumerate(feature_cnt)}
                # add "OTHER" class
                test_i = test_data.copy()
                # sample from other classes with the same distribution
                test_i[feature_name] = (
                    test_i[feature_name][np.array([freq_mapping[k] for k in test_i[feature_name]]) > top_n_categories]
                    .sample(n=test_data.shape[0], replace=True)
                    .values
                )
                preds = self.predict(test_i).data
                grid.append("<OTHER>")
                ys.append(preds)
                counts.append(feature_cnt.values[top_n_categories:].sum())
        # Datetime Features
        if self.reader._roles[feature_name].name == "Datetime":
            test_data_read = self.reader.read(test_data)
            feature_datetime = pd.arrays.DatetimeArray(test_data_read._data[feature_name])
            if datetime_level == "year":
                grid = np.unique([i.year for i in feature_datetime])
            elif datetime_level == "month":
                grid = np.arange(1, 13)
            else:
                grid = np.arange(7)
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = change_datetime(feature_datetime, datetime_level, i)
                preds = self.predict(test_i).data
                ys.append(preds)
            counts = Counter([getattr(i, datetime_level) for i in feature_datetime])
            counts = [counts[i] for i in grid]
        return grid, ys, counts

    def plot_pdp(
        self,
        test_data: ReadableToDf,
        feature_name: str,
        individual: Optional[bool] = False,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        top_n_classes: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
    ):
        grid, ys, counts = self.get_individual_pdp(
            test_data=test_data,
            feature_name=feature_name,
            n_bins=n_bins,
            top_n_categories=top_n_categories,
            datetime_level=datetime_level,
        )
        plot_pdp_with_distribution(
            test_data,
            grid,
            ys,
            counts,
            self.reader,
            feature_name,
            individual,
            top_n_classes,
            datetime_level,
        )


class TabularUtilizedAutoML(TimeUtilization):
    """Template to make TimeUtilization from TabularAutoML.

    Simplifies using ``TimeUtilization`` module for ``TabularAutoMLPreset``.

    Args:
        task: Task to solve.
        timeout: Timeout in seconds.
        memory_limit: Memory limit that are passed to each automl.
        cpu_limit: CPU limit that that are passed to each automl.
        gpu_ids: GPU IDs that are passed to each automl.
        timing_params: Timing params level that are passed to each automl.
        configs_list: List of str path to configs files.
        drop_last: Usually last automl will be stopped with timeout.
            Flag that defines if we should drop it from ensemble.
        return_all_predictions: skip blending phase
        max_runs_per_config: Maximum number of multistart loops.
        random_state: Initial random seed that will be set
            in case of search in config.

    """

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = None,
        timing_params: Optional[dict] = None,
        configs_list: Optional[Sequence[str]] = None,
        drop_last: bool = True,
        return_all_predictions: bool = False,
        max_runs_per_config: int = 5,
        random_state: int = 42,
        outer_blender_max_nonzero_coef: float = 0.05,
        **kwargs,
    ):
        if configs_list is None:
            configs_list = [
                os.path.join(_base_dir, "tabular_configs", x)
                for x in [
                    "conf_0_sel_type_0.yml",
                    "conf_1_sel_type_1.yml",
                    "conf_2_select_mode_1_no_typ.yml",
                    "conf_3_sel_type_1_no_inter_lgbm.yml",
                    "conf_4_sel_type_0_no_int.yml",
                    "conf_5_sel_type_1_tuning_full.yml",
                    "conf_6_sel_type_1_tuning_full_no_int_lgbm.yml",
                ]
            ]
        inner_blend = MeanBlender()
        outer_blend = WeightedBlender(max_nonzero_coef=outer_blender_max_nonzero_coef)
        super().__init__(
            TabularAutoML,
            task,
            timeout,
            memory_limit,
            cpu_limit,
            gpu_ids,
            timing_params,
            configs_list,
            inner_blend,
            outer_blend,
            drop_last,
            return_all_predictions,
            max_runs_per_config,
            None,
            random_state,
            **kwargs,
        )

    def get_feature_scores(
        self,
        calc_method: str = "fast",
        data: Optional[ReadableToDf] = None,
        features_names: Optional[Sequence[str]] = None,
        silent: bool = True,
    ):
        if calc_method == "fast":
            feat_imps = []
            for pipe in self.outer_pipes:
                for model in pipe.ml_algos:
                    fi = model.models[0][0].get_feature_scores("fast")
                    if fi is not None:
                        feat_imps.append(fi)
            n_feat_imps = len(feat_imps)
            if n_feat_imps == 0:
                if not silent:
                    logger.info2(
                        "No feature importances to show. " + "Please use another calculation method or another preset."
                    )
                return None
            return (
                pd.concat(feat_imps).groupby("Feature")["Importance"].agg(sum).sort_values(ascending=False)
                / n_feat_imps
            ).reset_index()

        if calc_method != "accurate":
            if not silent:
                logger.info2(
                    "Unknown calc_method. "
                    + "Currently supported methods for feature importances calculation are 'fast' and 'accurate'."
                )
            return None

        if data is None:
            if not silent:
                logger.info2("Data parameter is not setup for accurate calculation method. Aborting...")
            return None

        automl = self.outer_pipes[0].ml_algos[0].models[0][0]
        read_csv_params = automl._get_read_csv_params()
        data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)

        used_feats = set()
        for pipe in self.outer_pipes:
            used_feats.update(pipe.ml_algos[0].models[0][0].collect_used_feats())

        fi = calc_feats_permutation_imps(
            automl,
            list(used_feats),
            data,
            automl.reader.target,
            automl.task.get_dataset_metric(),
            silent=silent,
        )
        return fi

    def create_model_str_desc(self, pref_tab_num: int = 0, split_line_len: int = 80) -> str:
        res = "Final prediction for new objects = \n"
        for it, (model, weight) in enumerate(zip(self.outer_pipes, self.outer_blend.wts)):
            config_path = model.ml_algos[0].models[0][0].config_path.split("/")[-1]
            res += "\t" * (pref_tab_num + 1) + "+ " * (it > 0)
            res += '{:.5f} * {} averaged models with config = "{}" and different CV random_states. Their structures: \n\n'.format(
                weight, len(model.ml_algos[0].models[0]), config_path
            )
            for it1, m in enumerate(model.ml_algos[0].models[0]):
                cur_model_desc = m.create_model_str_desc(pref_tab_num + 2, split_line_len)
                res += "\t" * (pref_tab_num + 1) + "    Model #{}.\n{}\n\n".format(it1, cur_model_desc)

        return res

    def get_individual_pdp(
        self,
        test_data: ReadableToDf,
        feature_name: str,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
    ):
        reader = self.outer_pipes[0].ml_algos[0].models[0][0].reader
        assert feature_name in reader._roles
        assert datetime_level in ["year", "month", "dayofweek"]
        test_i = test_data.copy()
        # Numerical features
        if reader._roles[feature_name].name == "Numeric":
            counts, bin_edges = np.histogram(test_data[feature_name].dropna(), bins=n_bins)
            grid = (bin_edges[:-1] + bin_edges[1:]) / 2
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = i
                preds = self.predict(test_i).data
                ys.append(preds)
        # Categorical features
        if reader._roles[feature_name].name == "Category":
            feature_cnt = test_data[feature_name].value_counts()
            grid = list(feature_cnt.index.values[:top_n_categories])
            counts = list(feature_cnt.values[:top_n_categories])
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = i
                preds = self.predict(test_i).data
                ys.append(preds)
            if len(feature_cnt) > top_n_categories:
                freq_mapping = {feature_cnt.index[i]: i for i, _ in enumerate(feature_cnt)}
                # add "OTHER" class
                test_i = test_data.copy()
                # sample from other classes with the same distribution
                test_i[feature_name] = (
                    test_i[feature_name][np.array([freq_mapping[k] for k in test_i[feature_name]]) > top_n_categories]
                    .sample(n=test_data.shape[0], replace=True)
                    .values
                )
                preds = self.predict(test_i).data
                grid.append("<OTHER>")
                ys.append(preds)
                counts.append(feature_cnt.values[top_n_categories:].sum())
        # Datetime Features
        if reader._roles[feature_name].name == "Datetime":
            test_data_read = reader.read(test_data)
            feature_datetime = pd.arrays.DatetimeArray(test_data_read._data[feature_name])
            if datetime_level == "year":
                grid = np.unique([i.year for i in feature_datetime])
            elif datetime_level == "month":
                grid = np.arange(1, 13)
            else:
                grid = np.arange(7)
            ys = []
            for i in tqdm(grid):
                test_i[feature_name] = change_datetime(feature_datetime, datetime_level, i)
                preds = self.predict(test_i).data
                ys.append(preds)
            counts = Counter([getattr(i, datetime_level) for i in feature_datetime])
            counts = [counts[i] for i in grid]
        return grid, ys, counts

    def plot_pdp(
        self,
        test_data: ReadableToDf,
        feature_name: str,
        individual: Optional[bool] = False,
        n_bins: Optional[int] = 30,
        top_n_categories: Optional[int] = 10,
        top_n_classes: Optional[int] = 10,
        datetime_level: Optional[str] = "year",
    ):
        reader = self.outer_pipes[0].ml_algos[0].models[0][0].reader
        grid, ys, counts = self.get_individual_pdp(
            test_data=test_data,
            feature_name=feature_name,
            n_bins=n_bins,
            top_n_categories=top_n_categories,
            datetime_level=datetime_level,
        )
        plot_pdp_with_distribution(
            test_data,
            grid,
            ys,
            counts,
            reader,
            feature_name,
            individual,
            top_n_classes,
            datetime_level,
        )
