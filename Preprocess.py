import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocess():

    def __init__(self):
        print("Preprocess object created")

    def fillna(self, data, fill_strategies):
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print("{}: There is no such thing as preprocess strategy".format(strategy))

        return data

    def drop(self, data, drop_strategies):
        for column, strategy in drop_strategies.items():
            data = data.drop(labels=[column], axis=strategy)

        return data

    def feature_engineering(self, data, engineering_strategies=1):
        if engineering_strategies == 1:
            return self._feature_engineering1(data)

        return data

    def _feature_engineering1(self, data):

        data = self._base_feature_engineering(data)

        data['FareBin'] = pd.qcut(data['Fare'], 4)

        data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

        drop_strategy = {'Age': 1,  # 1 indicate axis 1(column)
                         'Name': 1,
                         'Fare': 1}
        data = self.drop(data, drop_strategy)

        return data

    def _base_feature_engineering(self, data):
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

        data['IsAlone'] = 1
        data.loc[(data['FamilySize'] > 1), 'IsAlone'] = 0

        data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split('.', expand=True)[0]
        min_length = 10
        title_names = (data['Title'].value_counts() < min_length)
        data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] is True else x)

        return data

    def _label_encoder(self, data):
        labelEncoder = LabelEncoder()
        for column in data.columns.values:
            if 'int64' == data[column].dtype or 'float64' == data[column].dtype or 'int64' == data[column].dtype:
                continue
            labelEncoder.fit(data[column])
            data[column] = labelEncoder.transform(data[column])
        return data

    def _get_dummies(self, data, prefered_columns=None):

        if prefered_columns is None:
            columns=data.columns.values
            non_dummies = None
        else:
            non_dummies = [col for col in data.columns.values if col not in prefered_columns]

            columns = prefered_columns

        dummies_data = [pd.get_dummies(data[col], prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)

class PreprocessStrategy():
    def __init__(self):
        self.data = None
        self._preprocessor = Preprocess()

    def strategy(self, data, strategy_type="strategy1"):
        self.data = data
        if strategy_type == 'strategy1':
            self._strategy1()
        elif strategy_type == 'strategy2':
            self._strategy2()

        return self.data

    def _base_strategy(self):
        drop_strategy = {'PassengerId': 1,  # 1 indicate axis 1(column)
                         'Cabin': 1,
                         'Ticket': 1}
        self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = {'Age': 'Median',
                         'Fare': 'Median',
                         'Embarked': 'Mode'}
        self.data = self._preprocessor.fillna(self.data, fill_strategy)

        self.data = self._preprocessor.feature_engineering(self.data, 1)

        self.data = self._preprocessor._label_encoder(self.data)

    def _strategy1(self):
        self._base_strategy()

        self.data = self._preprocessor._get_dummies(self.data,
                                                    prefered_columns=['Pclass', 'Sex', 'Parch',
                                                                      'Embarked', 'Title', 'IsAlone'])

    def _strategy2(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                                  prefered_columns=None) # None mean that all feature will be dummied