import os


result_folder_path = os.path.join(os.getcwd(), '..', '..', 'data', 'result')


class BaseIntegrate:
    def __init__(self) -> None:
        """
        Base class for integrating data
        result_folder_path: path to save result
        """
        self.result_folder_path = result_folder_path

    def integrate(self) -> None:
        """
        Integrate the data, feel free to open from processed_data_path
        Result should be saved to result_path, which is a folder, so name your own results
        Notice that the result should be in the same format as sample.csv
        """
        raise NotImplementedError