"""预测式卸载 CacheManager 的私有实现。"""

from .residency import PredictiveResidencyMixin
from .transfer import PredictiveTransferMixin

__all__ = ["PredictiveResidencyMixin", "PredictiveTransferMixin"]
