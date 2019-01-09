import matplotlib.pyplot as plt

from yellowbrick.features import RadViz


class Visualizer:

    def __init__(self):
        print("Visualizer object created!")

    def RandianViz(self, X, y, number_of_features):
        if number_of_features is None:
            features = X.columns.values
        else:
            features = X.columns.values[:number_of_features]

        fig, ax = plt.subplots(1, figsize=(15, 12))
        radviz = RadViz(classes=['survived', 'not survived'], features=features)

        radviz.fit(X, y)
        radviz.transform(X)
        radviz.poof()