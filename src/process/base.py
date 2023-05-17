import os


raw_data_path = os.path.join(os.getcwd(), '..', '..', 'data', 'raw', 'train.csv')
processed_data_path = os.path.join(os.getcwd(), '..', '..', 'data', 'processed', 'train.csv')


class BaseProcess:
    def __init__(self) -> None:
        """
        Base class for processing data
        raw_data_path: path to raw data's train.csv
        processed_data_path: path to processed data's train.csv
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def process(self) -> None:
        """
        Process the raw data, feel free to open from raw_data_path
        """
        raise NotImplementedError
    
    def save(self) -> None:
        """
        Save the processed data, feel free to save to processed_data_path
        """
        raise NotImplementedError
