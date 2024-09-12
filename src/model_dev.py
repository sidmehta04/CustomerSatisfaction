import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abcstract class for all models
    """
    def train(self,X_train,y_train):
        pass

class LinearRegModel(Model):
    def train(self,X_train,y_train,**kwargs):
        try:

            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training:{}".format(e))
            raise e
