import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass
class MSE(Evaluation):
    """
    Evaluation strategy using Mean Sqaured Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calc rmse {}".format(e))
            raise e


class R2(Evaluation):
    """
    Evaluation strategy using R-squared score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R-squared")
            r2 = r2_score(y_true, y_pred)
            logging.info("R-squared: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R-squared: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy using Root Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred,squared=False)
            rmse = np.sqrt(mse)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
