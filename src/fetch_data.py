import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class TitanicData():

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def fetch_data(self):
        """Fetch data from csv files."""
        self.data = pd.read_csv('../data/train.csv')


    def fill_age(self):
        """Fill null values in age with average by Sex, Embarked"""
        avg_age = self.data.groupby(['Sex', 'Embarked'])['Age'].transform('mean')
        self.data.loc[:, 'Age'] = np.where(
            self.data['Age'].isna(),
            avg_age,
            self.data['Age'].copy()
        )


    def make_categorical(self, colname):
        self.data = pd.get_dummies(data=self.data,
            prefix=colname,
            prefix_sep='_',
            dummy_na=False,
            columns=[colname],
            drop_first=True)


    def train_test_split(self, test_size=0.2):
        """Split self.data into training sets and test sets.

        Args:
            test_size (float, optional): Size of test set. Defaults to 0.2.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.data.drop('Survived', axis=1),
                np.array(self.data.loc[:, 'Survived']),
                test_size=test_size,
                random_state=123)


    def get_scale_cols(self):
        """Iterate through column names to build list scale_cols. To be
        used in self.scale.

        Returns:
            [list]: list of columns names to scale
        """

        scale_cols = ['Age', 'Fare']

        for col in self.data.columns:
            if ('Pclass' in col) | ('SibSp' in col) | ('Parch' in col):
                scale_cols.append(col)
        return scale_cols


    def scale(self):
        """Apply StandardScaler to columns with numerical values:
        'Age' and 'Fare'.
        
        """

        # scale_cols = ['Age', 'SibSp', 'Parch', 'Fare']
        scale_cols = self.get_scale_cols()
        sc = MinMaxScaler()

        self.X_train.loc[:, scale_cols] = \
            sc.fit_transform(self.X_train.loc[:, scale_cols].copy())

        self.X_test.loc[:, scale_cols] = \
            sc.transform(self.X_test.loc[:, scale_cols].copy())

