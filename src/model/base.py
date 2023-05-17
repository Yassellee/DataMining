import os


processed_data_path = os.path.join(os.getcwd(), '..', '..', 'data', 'processed', 'train.csv')
test_data_path = os.path.join(os.getcwd(), '..', '..', 'data', 'raw', 'test.csv')
result_path = os.path.join(os.getcwd(), '..', '..', 'data', 'result')


class BaseModel:
    def __init__(self) -> None:
        """
        Base class for model
        processed_data_path: path to processed data's train.csv
        test_data_path: path to raw data's test.csv
        result_path: path to save result
        """
        self.processed_data_path = processed_data_path
        self.test_data_path = test_data_path

    def fit(self) -> None:
        """
        Fit the model, feel free to open from processed_data_path
        """
        raise NotImplementedError
    
    def predict(self) -> None:
        """
        Predict the model, feel free to open from test_data_path
        Result should be saved to result_path, which is a folder, so name your own results
        Notice that the result should be in the same format as sample.csv
        """
        raise NotImplementedError