import copy
from typing import Any, Dict, Optional

import yaml


class CfgNode:
    """A very small subset of the yacs CfgNode API used by this project."""

    def __init__(self, init_dict: Optional[Dict[str, Any]] = None, *, new_allowed: bool = False):
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_new_allowed", bool(new_allowed))
        object.__setattr__(self, "_frozen", False)
        if init_dict:
            for key, value in init_dict.items():
                self._data[key] = self._wrap(value)

    # Internal helpers -----------------------------------------------------
    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict):
            return CfgNode(value, new_allowed=self._new_allowed)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def _assert_mutable(self) -> None:
        if self._frozen:
            raise AttributeError("Attempted to modify a frozen CfgNode")

    # Mapping-style interface ----------------------------------------------
    def merge_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration file {path} must define a mapping at the top level")
        self.merge_from_dict(loaded)

    def merge_from_dict(self, cfg_dict: Dict[str, Any]) -> None:
        for key, value in cfg_dict.items():
            self._merge_key(key, value)

    def _merge_key(self, key: str, value: Any) -> None:
        if key not in self._data:
            if not self._new_allowed:
                raise AttributeError(f"Key '{key}' is not in the config and new entries are disabled")
            self._data[key] = self._wrap(value)
            return

        current = self._data[key]
        if isinstance(current, CfgNode) and isinstance(value, dict):
            current.merge_from_dict(value)
        else:
            self._data[key] = self._wrap(value)

    # Attribute interface --------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_data", "_new_allowed", "_frozen"}:
            object.__setattr__(self, name, value)
            return
        self._assert_mutable()
        if not self._new_allowed and name not in self._data:
            raise AttributeError(f"Key '{name}' is not in the config and new entries are disabled")
        self._data[name] = self._wrap(value)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return repr(self._data)

    # Utility methods ------------------------------------------------------
    def clone(self) -> "CfgNode":
        return copy.deepcopy(self)

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, CfgNode):
                value.freeze()


CN = CfgNode
