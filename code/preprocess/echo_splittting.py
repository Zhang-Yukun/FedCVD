import pandas as pd


class ECHOSplitter:
    def __init__(self,
                 input_file: str,
                 train_output_file: str,
                 test_output_file: str,
                 test_sample_rate: float,
                 random_seed: int):
        self.input_file = input_file
        self.meta = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        self.test_sample_rate = test_sample_rate
        self.random_seed = random_seed

    def file_load(self):
        self.meta = pd.read_csv(self.input_file, dtype={'ECHO_ID': str})

    def file_save(self):
        self.train_data.sort_values(by=["ECHO_ID"], ascending=True, inplace=True)
        self.test_data.sort_values(by=["ECHO_ID"], ascending=True, inplace=True)

        self.train_data.to_csv(self.train_output_file, index=None, encoding="utf-8")
        self.test_data.to_csv(self.test_output_file, index=None, encoding="utf-8")

    def sample(self, data, sample_rate: float):
        selected_data = data.sample(frac=sample_rate, random_state=self.random_seed)
        unselected_data = data.drop(labels=selected_data.index)
        return unselected_data, selected_data

    def run(self):
        self.file_load()
        self.train_data, self.test_data = self.sample(self.meta, self.test_sample_rate)
        self.file_save()
