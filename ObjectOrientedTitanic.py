import pandas as pd

from Information import Information
from Preprocess import PreprocessStrategy
from Visualizer import Visualizer
from GridSearchHelper import GridSearchHelper


class ObjectOrientedTitanic():

    def __init__(self, train, test):
        print("ObjectOrientedTitanic object created")
        self.testPassengerID = test['PassengerId']
        self.number_of_train = train.shape[0]

        self.y_train = train['Survived']
        self.train = train.drop('Survived', axis=1)
        self.test = test

        self.all_data = self._get_all_data()

        # Create instance of objects
        self._info = Information()
        self.preprocessStrategy = PreprocessStrategy()
        self.visualizer = Visualizer()
        self.gridSearchHelper = GridSearchHelper()

    def _get_all_data(self):
        return pd.concat([self.train, self.test])

    def information(self):
        self._info.info(self.all_data)

    def preprocessing(self, strategy_type):
        self.strategy_type = strategy_type

        self.all_data = self.preprocessStrategy.strategy(self._get_all_data(), strategy_type)

    def visualize(self, visualizer_type, number_of_features=None):
        self._get_train_and_test()

        if visualizer_type == "RadViz":
            self.visualizer.RandianViz(X=self.X_train,
                                       y=self.y_train,
                                       number_of_features=number_of_features)

    def machine_learning(self):
        self._get_train_and_test()

        self.gridSearchHelper.fit_predict_save(self.X_train,
                                               self.X_test,
                                               self.y_train,
                                               self.testPassengerID,
                                               self.strategy_type)

    def _get_train_and_test(self):
        self.X_train = self.all_data[:self.number_of_train]
        self.X_test = self.all_data[self.number_of_train:]