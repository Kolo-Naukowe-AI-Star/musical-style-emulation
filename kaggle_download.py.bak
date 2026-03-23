import os

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset_slug, path="./data"):
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(path):
        os.makedirs(path)

    api.dataset_download_files(dataset_slug, path=path, unzip=True)


if __name__ == "__main__":
    target_dataset = "thepqwel/1989-taylor-audiocraft-dataset"  # for now
    download_kaggle_dataset(dataset_slug=target_dataset)
