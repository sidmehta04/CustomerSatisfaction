from zenml import pipeline
from steps.ingestion import ingest_data
from steps.clean_data import clean_data
from steps.eval import eval_model
from steps.model_train import model_train

@pipeline

def training_pipeline(data_path:str):
    df=ingest_data(data_path)
    clean_data(df)
    model_train(df)
    eval_model(df)


