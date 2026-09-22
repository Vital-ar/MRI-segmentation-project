import argparse

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


    from src.pipeline.model_search_v2 import model_search_v2 

    import lightning.pytorch as pl



    pl.seed_everything(RANDOM_STATE, workers=True)
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Run specific model training on TPU")
    
    # 2. Define the expected argument
    parser.add_argument(
        "--model_index", 
        type=int, 
        required=True, 
        help="Index of the model configuration to train"
    )
    
    # 3. Parse the command from the bash cell
    args = parser.parse_args()
    
    # 4. Pass the index into your pipeline
    model_search_v2(index=args.model_index)

if __name__ == "__main__":
    main()