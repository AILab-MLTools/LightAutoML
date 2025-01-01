"""Internal representation of dataset in numpy, pandas and csr formats."""

from copy import copy  # , deepcopy
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import Series
from scipy import sparse

from ..tasks.base import Task
from .base import IntIdx
from .base import LAMLDataset
from .base import RolesDict
from .base import array_attr_roles
from .base import valid_array_attributes
from .roles import ColumnRole
from .roles import DropRole
from .roles import NumericRole


# disable warnings later
# pd.set_option('mode.chained_assignment', None)

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]
DenseSparseArray = Union[np.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar("Dataset", bound=LAMLDataset)


# possible checks list
# valid shapes
# target var is ok for task
# pandas - roles for all columns are defined
# numpy - roles and features are ok each other
# numpy - roles and features are ok for data
# features names does not contain __ - it's used to split processing names

# sparse - do not replace init and set data, but move type assert in checks?


class NumpyDataset(LAMLDataset):
    """Dataset that contains info in np.ndarray format.

    Create dataset from numpy arrays.

    Args:
        data: 2d array of features.
        features: Features names.
        roles: Roles specifier.
        task: Task specifier.
        **kwargs: Named attributes like target, group etc ..

    Note:
        For different type of parameter feature there is different behavior:

            - list, should be same len as data.shape[1]
            - None - automatic set names like feat_0, feat_1 ...
            - Prefix - automatic set names like Prefix_0, Prefix_1 ...

        For different type of parameter feature there is different behavior:

            - list, should be same len as data.shape[1].
            - None - automatic set NumericRole(np.float32).
            - ColumnRole - single role.
            - dict.

    """

    # TODO: Checks here
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "NumpyDataset"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return list(self._features)

    @features.setter
    def features(self, val: Union[Sequence[str], str, None]):
        """Define how to set features.

        Args:
            val: Values of features.

        Note:
            There is different behavior for different type of val parameter:

                - list - should be same len as ``data.shape[1]``
                - None - automatic set names like `feat_0`, `feat_1` ...
                - `'Prefix'` - automatic set names
                  to `Prefix_0`, `Prefix_1` ...

        """
        if type(val) is list:
            self._features = copy(val)
        else:
            prefix = val if val is not None else "feat"
            self._features = ["{0}_{1}".format(prefix, x) for x in range(self.data.shape[1])]

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """Define how to set roles.

        Args:
            val: Roles.

        Note:
            There is different behavior for different type of val parameter:

                - `List` - should be same len as ``data.shape[1]``.
                - `None` - automatic set ``NumericRole(np.float32)``.
                - ``ColumnRole`` - single role for all.
                - ``dict``.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
        else:
            role = NumericRole(np.float32) if val is None else val
            self._roles = dict(((x, role) for x in self.features))

    def _check_dtype(self):
        """Check if dtype in ``.set_data`` is ok and cast if not.

        Raises:
            AttributeError: If there is non-numeric type in dataset.

        """
        # dtypes = list(set(map(lambda x: x.dtype, self.roles.values())))
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = np.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert np.issubdtype(self.dtype, np.number), "Support only numeric types in numpy dataset."

        if self.data.dtype != self.dtype:
            try:
                self.data = self.data.astype(self.dtype)
            except:
                pass

    def __init__(
        self,
        data: Optional[DenseSparseArray],
        features: NpFeatures = (),
        roles: NpRoles = None,
        task: Optional[Task] = None,
        **kwargs: np.ndarray
    ):

        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (), roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d np.ndarray of features.
            features: features names.
            roles: Roles specifier.

        Note:
            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        assert data is None or type(data) is np.ndarray, "Numpy dataset support only np.ndarray features"
        super().set_data(data, features, roles)
        self._check_dtype()

    @staticmethod
    def _hstack(datasets: Sequence[np.ndarray]) -> np.ndarray:
        """Concatenate function for numpy arrays.

        Args:
            datasets: Sequence of np.ndarray.

        Returns:
            Stacked features array.

        """
        return np.hstack(datasets)
    @staticmethod
    def _vstack(datasets: Sequence[np.ndarray]) -> np.ndarray:
        """Concatenate function for numpy arrays.

        Args:
            datasets: Sequence of np.ndarray.

        Returns:
            Stacked features array.

        """
        return np.vstack(datasets)

    @staticmethod
    def _get_rows(data: np.ndarray, k: IntIdx) -> np.ndarray:
        """Get rows slice for numpy ndarray.

        Args:
            data: Data.
            k: Sequence of integer indexes.

        Returns:
            Rows slice.

        """
        return data[k]

    @staticmethod
    def _get_cols(data: np.ndarray, k: IntIdx) -> np.ndarray:
        """Get cols slice.

        Args:
            data: Data.
            k: Sequence of integer indexes.

        Returns:
            Cols slice.

        """
        return data[:, k]

    @classmethod
    def _get_2d(cls, data: np.ndarray, k: Tuple[IntIdx, IntIdx]) -> np.ndarray:
        """Get 2d slice.

        Args:
            data: Data.
            k: Tuple of integer sequences.

        Returns:
            2d slice.

        """
        rows, cols = k

        return data[rows, cols]

    @staticmethod
    def _set_col(data: np.ndarray, k: int, val: np.ndarray):
        """Inplace set columns.

        Args:
            data: Data.
            k: Index of column.
            val: Values to set.

        """
        data[:, k] = val

    def to_numpy(self) -> "NumpyDataset":
        """Empty method to convert to numpy.

        Returns:
            Same NumpyDataset.

        """
        return self

    def to_csr(self) -> "CSRSparseDataset":
        """Convert to csr.

        Returns:
            Same dataset in CSRSparseDatatset format.

        """
        assert all(
            [self.roles[x].name == "Numeric" for x in self.features]
        ), "Only numeric data accepted in sparse dataset"
        data = None if self.data is None else sparse.csr_matrix(self.data)

        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return CSRSparseDataset(data, features, roles, task, **params)

    def to_pandas(self) -> "PandasDataset":
        """Convert to PandasDataset.

        Returns:
            Same dataset in PandasDataset format.

        """
        # check for empty case
        data = None if self.data is None else DataFrame(self.data, columns=self.features)
        roles = self.roles
        # target and etc ..
        params = dict(
            (
                (x, Series(self.__dict__[x]) if len(self.__dict__[x].shape) == 1 else DataFrame(self.__dict__[x]))
                for x in self._array_like_attrs
            )
        )
        task = self.task

        return PandasDataset(data, roles, task, **params)

    @staticmethod
    def from_dataset(dataset: Dataset) -> "NumpyDataset":
        """Convert random dataset to numpy.

        Args:
            dataset: Dataset.

        Returns:
            Numpy dataset.

        """
        return dataset.to_numpy()


class CSRSparseDataset(NumpyDataset):
    """Dataset that contains sparse features and np.ndarray targets."""

    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "CSRSparseDataset"

    @staticmethod
    def _get_cols(data: Any, k: Any):
        """Not implemented."""
        raise NotImplementedError

    @staticmethod
    def _set_col(data: Any, k: Any, val: Any):
        """Not implemented."""
        raise NotImplementedError

    def to_pandas(self) -> Any:
        """Not implemented."""
        raise NotImplementedError

    def to_numpy(self) -> "NumpyDataset":
        """Convert to NumpyDataset.

        Returns:
            NumpyDataset.

        """
        # check for empty
        data = None if self.data is None else self.data.toarray()
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.

        Returns:
            tuple of 2 elements.

        """
        rows, cols = None, None
        try:
            rows, cols = self.data.shape
        except TypeError:
            if len(self._array_like_attrs) > 0:
                rows = len(self.__dict__[self._array_like_attrs[0]])
        return rows, cols

    @staticmethod
    def _hstack(datasets: Sequence[Union[sparse.csr_matrix, np.ndarray]]) -> sparse.csr_matrix:
        """Concatenate function for sparse and numpy arrays.

        Args:
            datasets: Sequence of csr_matrix or np.ndarray.

        Returns:
            Sparse matrix.

        """
        return sparse.hstack(datasets, format="csr")
    def _vstack(datasets: Sequence[Union[sparse.csr_matrix, np.ndarray]]) -> sparse.csr_matrix:
        """Concatenate function for sparse and numpy arrays.

        Args:
            datasets: Sequence of csr_matrix or np.ndarray.

        Returns:
            Sparse matrix.

        """
        return sparse.vstack(datasets, format="csr")

    def __init__(
        self,
        data: Optional[DenseSparseArray],
        features: NpFeatures = (),
        roles: NpRoles = None,
        task: Optional[Task] = None,
        **kwargs: np.ndarray
    ):
        """Create dataset from csr_matrix.

        Args:
            data: csr_matrix of features.
            features: Features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Named attributes like target, group etc ..

        Note:
            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (), roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: csr_matrix of features.
            features: features names.
            roles: Roles specifier.

        Note:
            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        assert data is None or type(data) is sparse.csr_matrix, "CSRSparseDataset support only csr_matrix features"
        LAMLDataset.set_data(self, data, features, roles)
        self._check_dtype()

    @staticmethod
    def from_dataset(dataset: Dataset) -> "CSRSparseDataset":
        """Convert dataset to sparse dataset.

        Args:
            dataset: Dataset.

        Returns:
            Dataset in sparse form.

        """
        return dataset.to_csr()


class PandasDataset(LAMLDataset):
    """Dataset that contains `pd.DataFrame` features and `pd.Series` targets.

    Args:
        data: Table with features.
        features: features names.
        roles: Roles specifier.
        task: Task specifier.
        **kwargs: Series, array like attrs target, group etc...

    """

    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "PandasDataset"

    @property
    def features(self) -> List[str]:
        """Get list of features.

        Returns:
            list of features.

        """
        return [] if self.data is None else list(self.data.columns)

    @features.setter
    def features(self, val: None):
        """Ignore setting features.

        Args:
            val: ignored.

        """
        pass

    def __init__(
        self,
        data: Optional[DataFrame] = None,
        roles: Optional[RolesDict] = None,
        task: Optional[Task] = None,
        **kwargs: Series
    ):
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f].reset_index(drop=True)
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)

    def _get_cols_idx(self, columns: Union[Sequence[str], str]) -> Union[Sequence[int], int]:
        """Get numeric index of columns by column names.

        Args:
            columns: sequence of columns of single column.

        Returns:
            sequence of int indexes or single int.

        """
        if isinstance(columns, str):
            idx = self.data.columns.get_loc(columns)

        else:
            idx = self.data.columns.get_indexer(columns)

        return idx

    def set_data(self, data: DataFrame, features: None, roles: RolesDict):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `None`, just for same interface.
            roles: Dict with roles.

        """
        super().set_data(data, features, roles)
        self._check_dtype()

    def _check_dtype(self):
        """Check if dtype in .set_data is ok and cast if not."""
        date_columns = []

        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == "Datetime":
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        self.data = self.data.astype(self.dtypes)
        self.data.reset_index(drop=True, inplace=True)
        # do we need to reset_index ?? If yes - drop for Series attrs too
        # case to check - concat pandas dataset and from numpy to pandas dataset
        # TODO: Think about reset_index here
        # self.data.reset_index(inplace=True, drop=True)

        # handle dates types
        for i in date_columns:
            dt_role = self.roles[i]
            if not (self.data.dtypes[i] is np.datetime64):
                self.data[i] = pd.to_datetime(
                    self.data[i],
                    format=dt_role.format,
                    unit=dt_role.unit,
                    origin=dt_role.origin,
                    cache=True,
                )

            self.dtypes[i] = np.datetime64

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        return pd.concat(datasets, axis=1)
    
    @staticmethod
    def _vstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        return pd.concat(datasets, axis=0)

    @staticmethod
    def _get_rows(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """
        return data.iloc[k]

    @staticmethod
    def _get_cols(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get cols slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced cols.

        """
        return data.iloc[:, k]

    @classmethod
    def _get_2d(cls, data: DataFrame, k: Tuple[IntIdx, IntIdx]) -> FrameOrSeries:
        """Define 2d slice of table.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            2d sliced table.

        """
        rows, cols = k

        return data.iloc[rows, cols]

    @staticmethod
    def _set_col(data: DataFrame, k: int, val: Union[Series, np.ndarray]):
        """Inplace set column value to `pd.DataFrame`.

        Args:
            data: Table with data.
            k: Column index.
            val: Values to set.

        """
        data.iloc[:, k] = val

    def to_numpy(self) -> "NumpyDataset":
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """
        # check for empty
        data = None if self.data is None else self.data.values
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x].values) for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    def to_pandas(self) -> "PandasDataset":
        """Empty method, return the same object.

        Returns:
            Self.

        """
        return self

    @staticmethod
    def from_dataset(dataset: Dataset) -> "PandasDataset":
        """Convert random dataset to pandas dataset.

        Args:
            dataset: Dataset.

        Returns:
            Converted to pandas dataset.

        """
        return dataset.to_pandas()

    def nan_rate(self):
        """Counts overall number of nans in dataset.

        Returns:
            Number of nans.

        """
        return (len(self.data) - self.data.count()).sum()
