"""
Set of helpers for conviniently change layer functions inside function blocks
"""
from inspect import isclass, signature
import functools

import tensorflow.keras as keras
from tensorflow.keras.constraints import Constraint


class ConstraintWrapper:
    """Change all target layers' default constraint inside function block

    The implementation changes the function.__globals__ on the fly and restore
    it once the value is return. This is used as a decorator, ex:

    @ConstraintWrapper(MinMaxNorm, Conv2D, scope=globals())
    def target_funtion(...):
        pass

    then the Conv2D inside target function will all have a default MinMaxNorm
    constraint in constructor.
    """

    def __init__(self, constraint: Constraint, targets: set):
        """Create a ConstraintWrapper object

        Args:
            constraint (Constraint): the constraint to be assigned as defaults
            targets (Set[Layer]): the affected layers types
        """
        self._constraint = constraint

        msg = "target must be class type, got {}"
        for target in targets:
            assert isclass(target), msg.format(target)
        self._targets = set(targets)

    def __call__(self, target_func):

        @functools.wraps(target_func)
        def fn_with_mocked_layers(*args, **kwargs):
            kwords = {
                "kernel_constraint", "bias_constraint",
                "beta_constraint", "gamma_constraint"
            }
            stored_layers = {}
            scope = target_func.__globals__

            for key, item in scope.items():
                if isclass(item) and item in self._targets:

                    @functools.wraps(item)
                    def mocked_layer(
                            *args,
                            _layer_ref=item,
                            **kwargs
                            ):
                        argspec = set(signature(_layer_ref).parameters.keys())
                        kdefaults = {
                            kword: self._constraint for kword in kwords
                            if (kword in argspec) and (kword not in kwargs)
                        }
                        return _layer_ref(*args, **kdefaults, **kwargs)

                    stored_layers[key] = item
                    scope[key] = mocked_layer

            try:
                res = target_func(*args, **kwargs)
            finally:
                scope.update(stored_layers)
            return res

        return fn_with_mocked_layers
