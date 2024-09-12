import logging
import pandas as pd
from zenml import step
from src.eval import R2, RMSE, MSE
from sklearn.base import RegressorMixin
from typing import Tuple
import mlflow
from zenml.client import Client

from typing_extensions import Annotated

client = Client()
experiment_track = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_track.name if experiment_track else None)
def eval_model(model: RegressorMixin,
               X_test: pd.DataFrame,
               y_test: pd.DataFrame) -> Tuple[
                   Annotated[float, "r2_score"],
                   Annotated[float, "rmse"],
               ]:
    try:
        prediction = model.predict(X_test)
        mse_class = RMSE()
        mse = mse_class.calculate_score(y_test, prediction)
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)

        if experiment_track:
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

        return mse, r2
    except Exception as e:
        logging.error("Error in eval model: {}".format(e))
        raise e