"""
Set of helpers for conviniently change layer functions inside function blocks
"""
import inspect
import functools

import tensorflow.keras as keras
from tensorflow.keras.constraints import Constraint


class ConstraintWrapper:
    """Change all target layers' default constraint inside function block

    This is used as a decorator, ex:

    @ConstraintWrapper(MinMaxNorm, Conv2D, scope=globals())
    def target_funtion(...):
        pass

    then the Conv2D inside target function will all have a default MinMaxNorm
    constraint in constructor.
    """

    def __init__(
            self, constraint: Constraint, targets: set,
            scope_ref: dict
            ):
        """Create a ConstraintWrapper object

        Args:
            constraint (Constraint): the constraint to be assigned as defaults
            targets (Set[Layer]): the affected layers types
            scope_ref (dict):
                the variable scopes to mock, usually be the globals() of module
        """
        self._constraint = constraint

        msg = "target must be class type, got {}"
        for target in targets:
            assert inspect.isclass(target), msg.format(target)
        self._scope = scope_ref

    def __call__(self, target_class):

        @functools.wraps(target_class)
        def fn_with_mocked_layers(*args, **kwargs):
            stored_layers = {}

            for item in self._scope.values():
                if inspect.isclass(item):
                    print("inside wrap", item)

            for key, item in self._scope.items():
                if inspect.isclass(item) and item in self._targets:
                    @functools.wraps(item)
                    def mocked_layer(
                            *args,
                            kernel_constraint=self._constraint,
                            _layer_ref=item,
                            **kwargs
                            ):
                        return _layer_ref(
                            *args,
                            kernel_constraint=kernel_constraint,
                            **kwargs
                        )

                    stored_layers[key] = item
                    self._scope[key] = mocked_layer

            try:
                res = target_class(*args, **kwargs)
            finally:
                self._scope.update(stored_layers)
            return res

        return fn_with_mocked_layers
