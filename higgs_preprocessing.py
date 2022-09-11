from typing import List
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif


def feature_selection(data: pd.DataFrame, k: int) -> List[str]:
    X = data.drop("0", axis=1)
    y = data["0"]
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return selector.get_feature_names_out()


if __name__ == "__main__":
    # Main script for testing data pipelines

    train_data_1 = pd.read_csv(
        "./data/raw_data/higgs/training_part1.csv")
    train_data_2 = pd.read_csv(
        "./data/raw_data/higgs/training_part2.csv")
    train_data_3 = pd.read_csv(
        "./data/raw_data/higgs/training_part3.csv")

    train_data = pd.concat(
        [train_data_1, train_data_2, train_data_3], axis=0, ignore_index=True)

    test_data = pd.read_csv(
        "./data/raw_data/higgs/test.csv",
    )

    train_data.columns = [str(col) for col in train_data.columns]
    test_data.columns = [str(col) for col in test_data.columns]

    # preprocessing phase

    features_list_without_label = list(
        feature_selection(train_data, 25))

    features_list_with_label = features_list_without_label + ["0"]

    train_data = train_data[features_list_with_label].copy()
    test_data = test_data[features_list_with_label].copy()

    # train_data.iloc[:50].to_csv(
    #     "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_debug_training_1000.csv", index=False)
    # test_data.iloc[:20].to_csv(
    #     "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_debug_test_200.csv", index=False)

    train_data.to_csv(
        "./data/raw_data/higgs/fs_training.csv", index=False)
    test_data.to_csv(
        "./data/raw_data/higgs/fs_test.csv", index=False)
