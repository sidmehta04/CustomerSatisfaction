import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow 
from zenml.client import Client
experiment_track=Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_track.name if experiment_track else None)
def model_train(X_train:pd.DataFrame,
                X_test:pd.DataFrame,
                y_train:pd.Series,
                y_test:pd.Series,
                config:ModelNameConfig,
)->RegressorMixin:
    try:
        model=None
        if config.model_name=="LinearRegression":
            mlflow.sklearn.autolog()
            model=LinearRegModel() 
            trained_model=model.train(X_train,y_train)   
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error("Error in train model")
        raise e


    