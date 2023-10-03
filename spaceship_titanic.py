import pandas as pd

titance_df = pd.read_csv('train.csv')

numerical_column = [ 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Handle missing values by filling them with the mean of the respective column
for column in numerical_column:
    titance_df[column].fillna(titance_df[column].mean(), inplace=True)

# Remove duplicate rows based on all columns
titance_df.drop_duplicates(inplace=True)

# Perform Min-Max scaling for the specified column
for column in numerical_column:
    titance_df[column] = ((titance_df[column] - titance_df[column].min()) / 
                          (titance_df[column].max() - titance_df[column].min()))

# Perform one-hot encoding for the specified colum
titance_df = pd.get_dummies(
    titance_df, columns=['HomePlanet'], prefix=['HomePlanet']
)

# Create a new feature 'PassengerBilled' by adding 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
titance_df['PassengerBilled'] = (
    titance_df['RoomService'] +
    titance_df['FoodCourt'] +
    titance_df['ShoppingMall'] +
    titance_df['Spa'] +
    titance_df['VRDeck']
)

print(titance_df)
