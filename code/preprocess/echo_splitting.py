import pandas as pd
import os


def echo_split(meta: pd.DataFrame, sample_ratio: float = 0.8, seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the echo data into training set and validation set.

    Args:
        meta: The meta data of the echo data.
        sample_ratio: The ratio of the training set to the validation set.
        seed: The random seed.

    Returns:
        A tuple of two pandas.DataFrame objects, the first one is the training set, the second one is the validation set.
    """
    train_data = meta.sample(frac=sample_ratio, random_state=seed)
    test_data = meta.drop(labels=train_data.index)
    return train_data, test_data


if __name__ == "__main__":
    metadata = pd.read_csv("/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client1/metadata.csv")
    train_data, test_data = echo_split(metadata)
    train_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client1/train.csv",
        index=False, encoding="utf-8"
    )
    test_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client1/test.csv",
        index=False, encoding="utf-8"
    )
    metadata = pd.read_csv("/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client2/metadata.csv")
    train_data, test_data = echo_split(metadata)
    train_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client2/train.csv",
        index=False, encoding="utf-8"
    )
    test_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client2/test.csv",
        index=False, encoding="utf-8"
    )
    metadata = pd.read_csv("/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client3/metadata.csv")
    train_data, test_data = echo_split(metadata)
    train_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client3/train.csv",
        index=False, encoding="utf-8"
    )
    test_data.to_csv(
        "/Users/zhangyukun/Downloads/Datasets/ECHO/preprocessed/client3/test.csv",
        index=False, encoding="utf-8"
    )
