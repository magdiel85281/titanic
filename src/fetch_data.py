import pandas as pd 
import numpy as np


class TitanicData():

    def __init__(self):
        self.data = None


    def fetch_data(self):
        self.data = pd.read_csv('../data/train.csv')