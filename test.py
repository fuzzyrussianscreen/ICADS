
import pandas as pd
import csv

def test_answer():
    with open('facies_data.csv', newline='') as csvfile:
        dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))
        assert  len(dfSource) == 3233