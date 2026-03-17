"""
DistributionRegressor: Nonparametric distributional regression using LightGBM.
"""

from .distribution_regressor_CDF import DistributionRegressorCDF as DistributionRegressor
from .distribution_regressor_soft_target import DistributionRegressorSoftTarget
from .regressor import DistributionRegressor as DistributionRegressorLegacy

__version__ = "2.0.0"
__all__ = ["DistributionRegressor", "DistributionRegressorSoftTarget", "DistributionRegressorLegacy"]
