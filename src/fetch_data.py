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
        self.hidden_data = None


    def fetch_data(self):
        """Fetch data from csv files."""
        self.data = pd.read_csv('../data/train.csv')
        self.hidden_data = pd.read_csv('../data/test.csv')


    def fill_age(self):
        """Fill null values in age with average by Sex, Embarked"""
        avg_age = self.data.groupby(['Sex', 'Embarked'])['Age'].transform('mean')
        self.data.loc[:, 'Age'] = np.where(
            self.data['Age'].isna(),
            avg_age,
            self.data['Age'].copy()
        )


    def make_categorical(self, colname):
        """Apply pandas.get_dummies to colname and drop first.

        Parameters
        ----------
        colname : str
            name of column to make categorical
        """
        self.data = pd.get_dummies(data=self.data,
            prefix=colname,
            prefix_sep='_',
            dummy_na=False,
            columns=[colname],
            drop_first=True)


    def get_section(self):
        """Create column Section from char in Cabin."""
        self.data['Section'] = \
            np.where(
                self.data['Cabin'].notna(),
                self.data['Cabin'].astype(str).apply(lambda x: x[0]),
                np.nan
            )


    def add_cabin_flag(self):
        """Create column CabinFlag, 1 if Cabin is not null, else 0."""
        self.data.loc[:, 'CabinFlag'] = \
            np.where(self.data['Cabin'].isna(), 0, 1)


    def get_last_name(self):
        """Get last name from Name."""
        self.data['LastName'] = \
            self.data['Name'].apply(lambda x: x.split(',')[0])


class ModelInput():

    def __init__(self, data):
        self.data = data
        self.features = None
        self.model_input = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def set_feature_cols(self):
        """Iterate through column names to build list scale_cols. To be
        used in self.scale.

        Returns
        -------
        list
            list of columns names to scale
        """

        self.features = ['Age', 'Fare']

        for col in self.data.columns:
            if ('Pclass' in col) | ('Sex' in col) | ('SibSp' in col) | \
                ('Parch' in col) | ('Embarked' in col) | ('Section' in col):
                    self.features.append(col)
        
        self.features.append('Survived')


    def set_features(self):
        self.set_feature_cols()
        self.model_input = self.data.loc[:, self.features].copy()


    def train_test_split(self, test_size=0.2):
        """Split self.data into training sets and test sets.

        Parameters
        ----------
        test_size : float, optional
            Size of test set, by default 0.2
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.model_input.drop('Survived', axis=1),
                np.array(self.model_input.loc[:, 'Survived']),
                test_size=test_size,
                random_state=123)


    def scale(self):
        """Apply MinMaxScaler to model inputs."""
        sc = MinMaxScaler()

        self.X_train.loc[:, :] = sc.fit_transform(self.X_train)
        self.X_test.loc[:, :] = sc.transform(self.X_test)
