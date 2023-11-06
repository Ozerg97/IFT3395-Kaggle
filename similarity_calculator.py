import pandas as pd

class CSVComparator:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def compare(self):
        df1 = pd.read_csv(self.file1)
        df2 = pd.read_csv(self.file2)

        if df1.shape != df2.shape:
            raise ValueError("CSV files have different dimensions")

        similarity = (df1 == df2).mean().mean() * 100

        return similarity

file1 = 'sample_submission_og.csv'  # Replace with the path to your first CSV file
file2 = 'sample_submission_logreg.csv'  # Replace with the path to your second CSV file

comparator = CSVComparator(file1, file2)
similarity_percentage = comparator.compare()

print(f"The similarity between the two CSV files is: {similarity_percentage}%")
