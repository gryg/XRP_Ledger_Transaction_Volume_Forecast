"""Model implementations for XRP transaction volume forecasting."""

from src.models.base import TimeSeriesModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

__all__ = ["TimeSeriesModel", "XGBoostModel", "LSTMModel"]

# Optional imports that might not be available
try:
    from src.models.deepar_model import DeepARModel
    __all__.append("DeepARModel")
except ImportError:
    pass

try:
    from src.models.prophet_model import ProphetModel
    __all__.append("ProphetModel")
except ImportError:
    pass