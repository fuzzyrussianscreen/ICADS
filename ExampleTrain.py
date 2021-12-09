import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

with open('facies_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        print(str(row))#.split(","))
print(reader.line_num)
