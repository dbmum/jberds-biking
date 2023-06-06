from neural_network_model import NeuralNetworkModel
from .brian_model import BrianModel
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


def main():
    models: {NeuralNetworkModel: float} = {
        BrianModel(): 0
    }

    bike_rentals = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv")
    bike_rentals_holdout = pd.read_csv(
        "https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv")

    # We need to get the total rentals as the target regardless of preprocessing
    bike_rentals["total_rentals"] = bike_rentals.casual + bike_rentals.registered
    y = bike_rentals.total_rentals
    X = bike_rentals.drop(['target', 'casual', 'registered'], axis='columns')

    # Set aside a random test set immediately
    # X_train, X_test_master, y_train, y_test_master = train_test_split(X, y, test_size=0.2, random_state=0)

    # Set aside November as a master test set
    bike_rentals['date'] = pd.to_datetime(bike_rentals['dteday'], format='%d/%m/%y')
    split_date = datetime(2012, 11, 1)
    november_test_set = bike_rentals[bike_rentals['date'] >= split_date]
    train_set = bike_rentals[bike_rentals['date'] < split_date]


    for model, score in models.items():
        subset = X.copy(deep=True)
        model.train(subset)




if __name__ == "main":
    main()
