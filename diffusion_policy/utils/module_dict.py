from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator

import torch
from torch._jit_internal import _copy_to_script_wrapper

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self
from typing_extensions import deprecated
from torch.nn import Module

class ModuleDict:

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        self._modules = modules

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._modules.keys()


    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()


    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()
    
    def state_dict(self):
        """
        Returns the state dictionary for all optimizers.
        
        Returns:
            dict: A dictionary where the keys are the optimizer names and 
                  values are the corresponding optimizer state_dicts.
        """
        return {name: module.state_dict() for name, module in self._modules.items()}

    def load_state_dict(self, state_dicts):
        """
        Loads the state dictionary for each optimizer.
        
        Args:
            state_dicts (dict): A dictionary where the keys are optimizer names
                                and the values are the state_dicts to load for each optimizer.
        """
        if not isinstance(state_dicts, dict):
            raise ValueError("state_dicts must be a dictionary.")

        for name, state_dict in state_dicts.items():
            if name not in self._modules:
                raise ValueError(f"Module {name} is not in the ModuleDict.")
            
            self._modules[name].load_state_dict(state_dict)