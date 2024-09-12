import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivide,DataPreprocessingStrat
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],





]:
    try:
        preprocess_strat=DataPreprocessingStrat()
        data_clean=DataCleaning(df,preprocess_strat)
        processed_data=data_clean.handle_data()
        dvide_strat=DataDivide()
        data_clean=DataCleaning(processed_data,dvide_strat)
        X_train,X_test,y_train,y_test=data_clean.handle_data()
        logging.info("DataCleaning Done")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error in cleaning data {}".format(e))
        raise e
    
