"""Contains base classes for internal dataset interface."""

from copy import copy  # , deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from ..tasks.base import Task
from .roles import ColumnRole


valid_array_attributes = ("target", "group", "folds", "weights", "date", "id", "treatment")
array_attr_roles = ("Target", "Group", "Folds", "Weights", "Date", "Id", "Treatment")
# valid_tasks = ('reg', 'binary', 'multiclass') # TODO: Add multiclass and multilabel. Refactor for some dataset and pipes needed
# valid_tasks = ('reg', 'binary')


RolesDict = Dict[str, ColumnRole]
IntIdx = Union[Sequence[int], int]
RowSlice = Optional[Union[Sequence[int], Sequence[bool]]]
ColSlice = Optional[Union[Sequence[str], str]]


class LAMLColumn:
    """Basic class for pair - column, role."""

    def __init__(self, data: Any, role: ColumnRole):
        """Set a pair column/role.

        Args:
            data: 1d array like.
            role: Column role.

        """
        self.data = data
        self.role = role

    def __repr__(self) -> str:
        """Repr method.

        Returns:
            String with data representation.

        """
        return self.data.__repr__()


class LAMLDataset:
    """Basic class to create dataset."""

    # TODO: Create checks here
    _init_checks = ()  # list of functions that checks that _array_like_attrs are valid
    _data_checks = ()  # list of functions that checks that data in .set_data is valid for _array_like_attrs
    _concat_checks = ()  # list of functions that checks that datasets for concatenation are valid
    _dataset_type = "LAMLDataset"

    def __init__(
        self,
        data: Any,
        features: Optional[list],
        roles: Optional[RolesDict],
        task: Optional[Task] = None,
        **kwargs: Any
    ):
        """Create dataset with given data, features, roles and special attributes.

        Args:
            data: 2d array of data of special type for each dataset type.
            features: Feature names or None for empty data.
            roles: Features roles or None for empty data.
            task: Task for dataset if train/valid.
            **kwargs: Special named array of attributes (target, group etc.).

        """
        if features is None:
            features = []
        if roles is None:
            roles = {}
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def __len__(self):
        """Get count of rows in dataset.

        Returns:
            Number of rows in dataset.

        """
        return self.shape[0]

    def __repr__(self):
        """Get str representation.

        Returns:
            String with data representation.

        """
        # TODO: View for empty
        return self.data.__repr__()

    # default behavior and abstract methods
    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        """Select a subset of dataset.

        Define how to slice a dataset
        in way ``dataset[[1, 2, 3...], ['feat_0', 'feat_1'...]]``.
        Default behavior based on ``._get_cols``, ``._get_rows``, ``._get_2d``.

        Args:
            k: First element optional integer columns indexes,
                second - optional feature name or list of features names.

        Returns:
            Subdataset.

        """
        # TODO: Maybe refactor this part?
        if type(k) is tuple:
            rows, cols = k
        else:
            rows = k
            cols = None

        # case when columns are defined
        if cols is not None:
            idx = self._get_cols_idx(cols)
            data = self._get_2d(self.data, (rows, idx))

            # case of single column - return LAMLColumn
            if isinstance(cols, str):
                dataset = LAMLColumn(self._get_2d(self.data, (rows, idx)), role=self.roles[cols])

                return dataset

            # case of multiple columns - return LAMLDataset
            roles = dict(((x, self.roles[x]) for x in self.roles if x in cols))
            features = [x for x in cols if x in set(self.features)]
        else:
            data, roles, features = self.data, self.roles, self.features

        # case when rows are defined
        if rows is None:
            dataset = self.empty()
        else:
            dataset = copy(self)
            params = dict(((x, self._get_rows(self.__dict__[x], rows)) for x in self._array_like_attrs))
            dataset._initialize(self.task, **params)
            data = self._get_rows(data, rows)

        dataset.set_data(data, features, roles)

        return dataset

    def __setitem__(self, k: str, val: Any):
        """Inplace set values for single column (in default implementation).

        Args:
            k: Feature name.
            val: :class:`~lightautoml.dataset.base.LAMLColumn`
                or 1d array like.

        """
        assert k in self.features, "Can only replace existed columns in default implementations."
        idx = self._get_cols_idx(k)
        # for case when setting col and change role
        if type(val) is LAMLColumn:
            assert val.role.dtype == self.roles[k].dtype, "Inplace changing types unavailable."
            self._set_col(self.data, idx, val.data)
            self.roles[k] = val.role
        # for case only changing column values
        else:
            self._set_col(self.data, idx, val)

    def __getattr__(self, item: str) -> Any:
        """Get item for key features as target/folds/weights etc.

        Args:
            item: Attribute name.

        Returns:
            Attribute value.

        """
        if item in valid_array_attributes:
            return None
        raise AttributeError

    @property
    def features(self) -> list:
        """Define how to get features names list.

        Returns:
            Features names.

        """
        return list(self._features)

    @features.setter
    def features(self, val: list):
        """Define how to set features list.

        Args:
            val: Features names.

        """
        self._features = copy(val)

    @property
    def data(self) -> Any:
        """Get data attribute.

        Returns:
            Any, array like or ``None``.

        """
        return self._data

    @data.setter
    def data(self, val: Any):
        """Set data array or ``None``.

        Args:
            val: Some data or ``None``.

        """
        self._data = val

    @property
    def roles(self) -> RolesDict:
        """Get roles dict.

        Returns:
            Dict of feature roles.

        """
        return copy(self._roles)

    @roles.setter
    def roles(self, val: RolesDict):
        """Set roles dict.

        Args:
            val: Roles dict.

        """
        self._roles = dict(((x, val[x]) for x in self.features))

    @property
    def inverse_roles(self) -> Dict[ColumnRole, List[str]]:
        """Get inverse dict of feature roles.

        Returns:
            dict, keys - roles, values - features names.

        """
        inv_roles = {}

        roles = self.roles

        for k in roles:
            r = roles[k]
            if r in inv_roles:
                inv_roles[r].append(k)
            else:
                inv_roles[r] = [k]

        return inv_roles

    def _initialize(self, task: Optional[Task], **kwargs: Any):
        """Initialize empty dataset with task and array like attributes.

        Args:
            task: Task name for dataset.
            **kwargs: 1d arrays like attrs like target, group etc.

        """
        assert all([x in valid_array_attributes for x in kwargs]), "Unknown array attribute. Valid are {0}".format(
            valid_array_attributes
        )

        self.task = task
        # here we set target and group and so ...
        self._array_like_attrs = []
        for k in kwargs:
            self._array_like_attrs.append(k)
            self.__dict__[k] = kwargs[k]

        # checks for valid values in target, groups ...
        for check in self._init_checks:
            check(self)

        # set empty attributes
        self._data = None
        self._features = []
        self._roles = {}

    def set_data(self, data: Any, features: Any, roles: Any):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d array like or ``None``.
            features: List of features names.
            roles: Roles dict.

        """
        self.data = data
        self.features = features
        self.roles = roles

        # data checks
        for check in self._data_checks:
            check(self)

    def empty(self) -> "LAMLDataset":
        """Get new dataset for same task and targets, groups, without features.

        Returns:
            New empty dataset.

        """
        dataset = copy(self)
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        dataset._initialize(self.task, **params)

        return dataset

    def _get_cols_idx(self, columns: Sequence) -> Union[List[int], int]:
        """Get numeric index of columns by column names.

        Args:
            columns: Features names.

        Returns:
            List of integer indexes of single int.

        """
        if isinstance(columns, str):
            idx = self.features.index(columns)

        else:
            idx = [self.features.index(x) for x in columns]

        return idx

    # default calculated properties
    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.

        Returns:
            Tuple of 2 elements.

        """
        rows, cols = None, None
        try:
            rows, cols = len(self.data), len(self.features)
        except TypeError:
            if len(self._array_like_attrs) > 0:
                rows = len(self.__dict__[self._array_like_attrs[0]])
        return rows, cols

    # static methods - how to make 1d slice, 2s slice, concat of feature matrix etc ...
    @staticmethod
    def _vstack(datasets: Sequence[Any]) -> Any:
        """Abstract method - define horizontal stack of feature arrays.

        Args:
            datasets: Sequence of feature arrays.

        Returns:  # noqa DAR202
            Single feature array.

        """
        raise NotImplementedError("Horizontal Stack not implemented.")
    
    @staticmethod
    def _hstack(datasets: Sequence[Any]) -> Any:
        """Abstract method - define horizontal stack of feature arrays.

        Args:
            datasets: Sequence of feature arrays.

        Returns:  # noqa DAR202
            Single feature array.

        """
        raise NotImplementedError("Horizontal Stack not implemented.")

    @staticmethod
    def _get_rows(data, k: IntIdx) -> Any:
        """Abstract - define how to make rows slice of feature array.

        Args:
            data: 2d feature array.
            k: Sequence of int indexes or int.

        Returns:  # noqa DAR202
            2d feature array.

        """
        raise NotImplementedError("Row Slice not Implemented.")

    @staticmethod
    def _get_cols(data, k: IntIdx) -> Any:
        """Abstract - define how to make columns slice of feature array.

        Args:
            data: 2d feature array.
            k: Sequence indexes or single index.

        Returns:  # noqa DAR202
            2d feature array.

        """
        raise NotImplementedError("Column Slice not Implemented.")

    # TODO: remove classmethod here ?
    @classmethod
    def _get_2d(cls, data: Any, k: Tuple[IntIdx, IntIdx]) -> Any:
        """Default implementation of 2d slice based on rows slice and columns slice.

        Args:
            data: 2d feature array.
            k: Tuple of integer sequences or 2 int.

        Returns:
            2d feature array.

        """
        rows, cols = k

        return cls._get_rows(cls._get_cols(data, cols), rows)

    @staticmethod
    def _set_col(data: Any, k: int, val: Any):
        """Abstract - set a value of single column by column name inplace.

        Args:
            data: 2d feature array.
            k: Column idx.
            val: 1d column value.

        """
        raise NotImplementedError("Column setting inplace not implemented.")

    @classmethod
    def concat(cls, datasets: Sequence["LAMLDataset"]) -> "LAMLDataset":
        """Concat multiple dataset.

        Default behavior - takes empty dataset from datasets[0]
        and concat all features from others.

        Args:
            datasets: Sequence of datasets.

        Returns:
            Concated dataset.

        """
        for check in cls._concat_checks:
            check(datasets)

        dataset = datasets[0].empty()
        data = []
        features = []
        roles = {}

        atrs = set(dataset._array_like_attrs)
        for ds in datasets:
            data.append(ds.data)
            features.extend(ds.features)
            roles = {**roles, **ds.roles}
            for atr in ds._array_like_attrs:
                if atr not in atrs:
                    dataset._array_like_attrs.append(atr)
                    dataset.__dict__[atr] = ds.__dict__[atr]
                    atrs.update({atr})

        data = cls._hstack(data)
        dataset.set_data(data, features, roles)

        return dataset
    @classmethod
    def vconcat(cls, datasets: Sequence["LAMLDataset"]) -> "LAMLDataset":
        """Concat multiple dataset.

        Default behavior - takes empty dataset from datasets[0]
        and concat all features from others.

        Args:
            datasets: Sequence of datasets.

        Returns:
            Concated dataset.

        """
        for check in cls._concat_checks:
            check(datasets)

        dataset = datasets[0].empty()
        data = []
        features = [*datasets[0].features]
        roles = {**datasets[0].roles}
        atrs = set(dataset._array_like_attrs)

        for ds in datasets:
            data.append(ds.data)
            for atr in ds._array_like_attrs:
                if atr not in atrs:
                    dataset._array_like_attrs.append(atr)
                    dataset.__dict__[atr] = ds.__dict__[atr]
                    atrs.update({atr})

        data = cls._vstack(data)
        dataset.set_data(data, features, roles)
    
        return dataset
    
    def drop_features(self, droplist: Sequence[str]):
        """Inplace drop columns from dataset.

        Args:
            droplist: Feature names.

        Returns:
            Dataset without columns.

        """
        if len(droplist) == 0:
            return self
        return self[:, [x for x in self.features if x not in droplist]]

    @staticmethod
    def from_dataset(dataset: "LAMLDataset") -> "LAMLDataset":
        """Abstract method - how to create this type of dataset from others.

        Args:
            dataset: Original type dataset.

        Returns:  # noqa DAR202
            Converted type dataset.

        """
        raise NotImplementedError

    @property
    def dataset_type(self):
        """Get type of dataset."""
        return self._dataset_type
