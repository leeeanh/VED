from .abstract.hookcatalog import HookCatalog
from .base import get_base_hooks
from .comem_hooks import get_comem_hooks

from .hooks_api import HookAPI

__all__ = ['HookAPI']

def register_hooks():
    HookCatalog.register('base.VisScoreHook', lambda name:get_base_hooks(name))
    HookCatalog.register('base.TSNEHook', lambda name:get_base_hooks(name))
    HookCatalog.register('comem.COMemAEEvaluateHook', lambda name:get_comem_hooks(name))


register_hooks()
