"""
DistributionRegressor: Nonparametric distributional regression using LightGBM.
"""

from .distribution_regressor_soft_target import DistributionRegressorSoftTarget as DistributionRegressor
from .regressor import DistributionRegressor as DistributionRegressorLegacy

__version__ = "1.2.0"
__all__ = ["DistributionRegressor", "DistributionRegressorLegacy"]
