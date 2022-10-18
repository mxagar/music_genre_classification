# Reproducible ML Pipelines: Utilities

This folder contains an auxiliary python package for the project.
In it objects and functions used in different components/steps are defined.

For now, only the module `transformations.py` has been added.
This module contains custom feature transformers derived from `sklearn`.

If more modules were added, we would add the new functions, variables and classes to `__init__.py`.

To use this package/library in a component below the root level:

```python
# Add root path so that utilities package is found (for custom transformations)
sys.path.insert(1, '..')
from util_lib import ModeImputer
```
