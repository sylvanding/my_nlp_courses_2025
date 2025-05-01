from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


def load_training_dataset(
    dataset_name: str = "espejelomar/code_search_net_python_10000_examples",
    split: str = "train",
) -> DatasetDict:
    dataset = load_dataset(dataset_name, split=split).filter(
        lambda x: x["func_code_url"].startswith("https://github.com/apache")
    )
    return dataset


if __name__ == "__main__":
    dataset = load_training_dataset()
    print(dataset)
