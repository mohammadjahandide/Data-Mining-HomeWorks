import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

######################### HW1 #########################
print('######################### HW1 #########################')
titanic_df = pd.read_csv('train.csv')

numerical_column = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Handle missing values by filling them with the mean of the respective column
for column in numerical_column:
    titanic_df[column].fillna(titanic_df[column].mean(), inplace=True)

# Remove duplicate rows based on all columns
titanic_df.drop_duplicates(inplace=True)

# Perform Min-Max scaling for the specified column
for column in numerical_column:
    titanic_df[column] = ((titanic_df[column] - titanic_df[column].min()) /
                          (titanic_df[column].max() - titanic_df[column].min()))

# Perform one-hot encoding for the specified colum
titanic_df = pd.get_dummies(
    titanic_df, columns=['HomePlanet'], prefix=['HomePlanet']
)

# Create a new feature 'PassengerBilled' by adding 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
titanic_df['PassengerBilled'] = (
        titanic_df['RoomService'] +
        titanic_df['FoodCourt'] +
        titanic_df['ShoppingMall'] +
        titanic_df['Spa'] +
        titanic_df['VRDeck']
)

print(titanic_df)

######################### HW2 #########################
print('\n\n######################### HW2 #########################')
# Extract the target variable 'Survived' and print class distribution before undersampling
y_titanic = titanic_df['Transported']
print("Class distribution before undersampling:")
print(y_titanic.value_counts())
# Features (X) and target variable (y)
X = titanic_df.drop(columns=['Transported'])
y = titanic_df['Transported']
# Construct RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
# Apply random undersampling to the entire dataset
X_undersampled, y_undersampled = undersampler.fit_resample(X, y)
# Verify the class distribution after undersampling
print("Class distribution after undersampling:")
print(y_undersampled.value_counts())
