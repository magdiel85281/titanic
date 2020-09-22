import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def fill_age(df):
    """Fill null values in Age with average by Sex, Embarked.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data
    """
    avg_age = df.groupby(['Sex', 'Embarked'])['Age'].transform('mean')
    df.loc[:, 'Age'] = np.where(
        df['Age'].isna(),
        avg_age,
        df['Age'].copy()
    )


def get_last_name(df):
    """Get last name from Name.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data
    """

    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0])


def get_section(df):
    """Create column Section from char in Cabin.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data
    """

    df['Section'] = np.where(
            df['Cabin'].notna(),
            df['Cabin'].astype(str).apply(lambda x: x[0]),
            np.nan
        )


def add_cabin_flag(df):
    """Create column CabinFlag, 1 if Cabin is not null, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data
    """

    df.loc[:, 'CabinFlag'] = np.where(df['Cabin'].isna(), 0, 1)


def make_categorical(df, colname):
    """Apply pandas.get_dummies to colname and drop first.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data
    colname : str
        name of column to make categorical

    Returns
    -------
    pd.DataFrame
        Titanic data with dummied column
    """

    df = pd.get_dummies(data=df,
        prefix=colname,
        prefix_sep='_',
        dummy_na=False,
        columns=[colname],
        drop_first=True)
    return df


def get_all_dummies(df):
    """Convert all categorical columns into flags by value.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic data

    Returns
    -------
    pd.DataFrame
        Titanic data with dummied columns
    """

    col_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Section']
    for col in col_list:
        df = make_categorical(df, col)
    return df



class TitanicData():

    def __init__(self):
        self.data = None
        self.test_data = None


    def fetch_data(self):
        """Fetch data from csv files."""
        self.data = pd.read_csv('../data/train.csv')
        self.test_data = pd.read_csv('../data/test.csv')


    def set(self):
        """Execute sequence of functions to prepare data for model."""
        for df in [self.data, self.test_data]:
            fill_age(df)
            get_last_name(df)
            get_section(df)
            add_cabin_flag(df)
        self.data = get_all_dummies(self.data)
        self.test_data = get_all_dummies(self.test_data)



class ModelInput():

    def __init__(self, data, test_data):
        self.data = data
        self.test_data = test_data
        self.features = None
        self.model_input = None
        self.hidden_data = None
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

        # self.features.append('Survived')


    def set_features(self):
        """Set list of features and input for modelling."""
        self.set_feature_cols()
        self.model_input = self.data.loc[:, self.features].copy()

        # set hidden data to use the same features, filling nulls with 0
        self.hidden_data = self.test_data.reindex(
            columns=self.features).copy().fillna(0)


    def train_test_split(self, test_size=0.2):
        """Split self.data into training sets and test sets.

        Parameters
        ----------
        test_size : float, optional
            Size of test set, by default 0.2
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.model_input.copy(),
                np.array(self.data.loc[:, 'Survived']),
                test_size=test_size,
                random_state=123)


    def scale(self):
        """Apply MinMaxScaler to model inputs."""
        sc = MinMaxScaler()

        self.X_train.loc[:, :] = sc.fit_transform(self.X_train)
        self.X_test.loc[:, :] = sc.transform(self.X_test)
        self.hidden_data.loc[:, :] = sc.transform(self.hidden_data)
