import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

titanic_df = pd.read_csv('train.csv')
titanic_df['PassengerBilled'] = (
        titanic_df['RoomService'] +
        titanic_df['FoodCourt'] +
        titanic_df['ShoppingMall'] +
        titanic_df['Spa'] +
        titanic_df['VRDeck']
)
# Histogram for PassengerBilled distribution
plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['PassengerBilled'], bins=40, kde=True)
plt.title('PassengerBilled Distribution')
plt.xlabel('PassengerBilled')
plt.ylabel('Count')
plt.show()
# Bar plot for categorical data, for example, 'HomePlanet'
plt.figure(figsize=(6, 4))
sns.countplot(x='HomePlanet', data=titanic_df)
plt.title('HomePlanet Distribution')
plt.xlabel('HomePlanet')
plt.ylabel('Count')
plt.show()

# Calculate mean, median, and standard deviation for 'PassengerBilled'
mean_passenger_billed = titanic_df['PassengerBilled'].mean()
median_passenger_billed = titanic_df['PassengerBilled'].median()
std_dev_passenger_billed = titanic_df['PassengerBilled'].std()
# Print the calculated statistics
print(f'Mean PassengerBilled: {mean_passenger_billed}')
print(f'Median PassengerBilled: {median_passenger_billed}')
print(f'Standard Deviation of PassengerBilled: {std_dev_passenger_billed}')

# Correlation matrix to understand feature relationships
correlation_matrix = titanic_df.corr(numeric_only=True)
# Print the correlation matrix
print(correlation_matrix)
# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
