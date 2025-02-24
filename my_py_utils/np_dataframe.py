import numpy as np


class NumpyFrame(np.ndarray):
    """
    A 2D numpy array with column names.
    """

    def __new__(cls, data, columns: list[str]):
        obj = np.asarray(data).view(cls)

        if obj.ndim != 2:
            raise ValueError("Input array must be 2D")

        if len(columns) != obj.shape[1]:
            raise ValueError(f"Length of columns ({len(columns)}) does not match array width ({obj.shape[1]})")
        obj.columns = list(columns)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.columns = getattr(obj, 'columns', None)

    def __getitem__(self, key):
        # Handle list of column names
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            col_indices = [self.columns.index(col) for col in key if col in self.columns]
            result = super().__getitem__((slice(None), col_indices))
            result = result.view(NumpyFrame)
            result.columns = key
            return result

        # Handle single column name
        if isinstance(key, str):
            if key not in self.columns:
                raise KeyError(f"Column '{key}' not found")
            col_idx = self.columns.index(key)
            return super().__getitem__((slice(None), col_idx))

        # Handle integer indexing for rows
        if isinstance(key, int):
            result = super().__getitem__(key)
            if isinstance(result, np.ndarray) and result.ndim == 1:
                return result

        # Handle slicing and other indexing
        result = super().__getitem__(key)
        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                if isinstance(key, tuple) and len(key) > 1:
                    if isinstance(key[1], (slice, list, np.ndarray)):
                        if isinstance(key[1], slice):
                            new_columns = self.columns[key[1]]
                        else:
                            new_columns = [self.columns[i] for i in key[1]]
                        result = result.view(NumpyFrame)
                        result.columns = new_columns
                else:
                    result = result.view(NumpyFrame)
                    result.columns = self.columns
        return result

    def select(self, columns: list[str]) -> 'NumpyFrame':
        """Select specific columns by name."""
        return self[columns]

    def to_numpy(self) -> np.ndarray:
        return self.view(np.ndarray)
