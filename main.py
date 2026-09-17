import os
from dotenv import load_dotenv

def main():

    load_dotenv()

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None) is not None:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["MLFLOW_TRACKING_URI"] = user_secrets.get_secret("MLFLOW_TRACKING_URI")
        os.environ["DAGSHUB_USERNAME"] = user_secrets.get_secret("DAGSHUB_USERNAME")
        os.environ["DAGSHUB_TOKEN"] = user_secrets.get_secret("DAGSHUB_TOKEN")

        from src.constants.kaggle import RANDOM_STATE

    else:
        from src.constants.local import RANDOM_STATE


    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN")

    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("DAGSHUB_TOKEN")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("DAGSHUB_TOKEN")


    from src.pipeline.model_selction_pipe import run_model_selection_pipe

    import lightning.pytorch as pl



    pl.seed_everything(RANDOM_STATE, workers=True)

    run_model_selection_pipe()

if __name__ == '__main__':

    main()
