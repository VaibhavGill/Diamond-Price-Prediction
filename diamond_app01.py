import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 16,7
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/diamonds.csv')
df.head()

df.info()

df.isna().sum()

df.drop(axis = 1 , index = 0 , columns = 'Unnamed: 0' , inplace = True)

df.head()

# Boxplot for 'cut' vs 'price'
plt.title('Boxplot between "CUT" and "PRICE".')
sns.boxplot(x = df['cut'] , y = df['price'])

# Boxplot for 'color' vs 'price'
plt.title('Boxplot between "COLOR" and "PRICE".')
sns.boxplot(x = df['color'] , y = df['price'])

# Boxplot for 'clarity' vs 'price'
plt.title('Boxplot between "CLARITY" and "PRICE".')
sns.boxplot(x = df['clarity'] , y = df['price'])

# Create scatter plot with 'carat' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "CARAT" and "PRICE".')
plt.scatter(df['carat'] , df['price'] , color = '#3875E3' , marker = '*')

# Create scatter plot with 'depth' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "Depth" and "PRICE".')
plt.scatter(df['depth'] , df['price'] , color = '#A038E3' , marker = '+')

# Create scatter plot with 'table' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "Table" and "PRICE".')
plt.scatter(df['table'] , df['price'] , color = '#38E37A' , marker = '+')

# Create scatter plot with attribute 'x' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "X" and "PRICE".')
plt.scatter(df['x'] , df['price'] , color = '#C13481' , marker = '*')

# Create scatter plot with attribute 'y' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "Y" and "PRICE".')
plt.scatter(df['y'] , df['price'] , color = '#C13443' , marker = '*')

# Create scatter plot with 'z' on X-axis and 'price' on Y-axis
plt.title('Scatter plot between "Z" and "PRICE".')
plt.scatter(df['z'] , df['price'] , color = '#C18534' , marker = '*')

# Create a normal distribution curve for the `price`.
p = df['price']
plt.title('Normal Distribution Curve for the "price".')
sns.distplot(df['price'], bins = 'sturges' , hist = False , color = '#34C1B7')
plt.axvline(p.mean(),label = f'Mean price of diamonds is = {p.mean():.3f}' , color = '#000000')
plt.legend()
plt.show()
# Create a probablity density function for plotting the normal distribution
def prob(arr,mean,std):
  coeff = 1/(std*np.sqrt(2*np.pi))
  power = np.exp(-((arr-mean)**2/(2*(std**2))))
  return coeff*power
# Plot the normal distribution curve using plt.scatter()
rho = prob(df["price"].sort_values(),df["price"].mean(),df["price"].std())
plt.title('Normal Distribution curve using "plt.scatter()"')
plt.scatter(df["price"].sort_values(),rho)
plt.axvline(x=df["price"].mean(),label="mean of price" , color = '#000000')
plt.legend()
plt.show()

# Replace values of 'cut' column
df["cut"].replace({"Fair": 1, "Good": 2 , "Very Good" : 3 , "Premium" : 4 , "Ideal" : 5}, inplace=True)
df.head()

# Replace values of 'color' column
df["color"].replace({"D": 1, "E": 2 , "F" : 3 , "G" : 4 , "H" : 5 , "I" : 6 , "J" : 7}, inplace=True)
df.head()


# Replace values of 'clarity' column
df["clarity"].replace({"I1": 1, "SI2": 2 , "SI1" : 3 , "VS2" : 4 , "VS1" : 5 , "VVS2" : 6 , "VVS1" : 7 , "IF" : 8}, inplace=True)
df

# Create a list of feature variables.
features = list(df.columns)
features.remove('price')
features

# Build multiple linear regression model using all the features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the DataFrame into the train and test sets such that test set has 33% of the values.
X = df[features]
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X , Y , test_size = 0.33 , random_state = 42)

# Build linear regression model using the 'sklearn.linear_model' module.
linear_regression = LinearRegression()
linear_regression.fit(X_train , y_train)

# Print the value of the intercept
print("\nConstant".ljust(15, " "), f"{linear_regression.intercept_:.6f}")

# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, linear_regression.coef_)):
  print(f"{i[0]}".ljust(15, " "), f"{i[1]:.6f}")
  
  
  # Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_train_pred = linear_regression.predict(X_train)
y_test_pred = linear_regression.predict(X_test)

print(f"Train Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_train, y_train_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.3f}")

print(f"\n\nTest Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_test, y_test_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.3f}")

# Heatmap to pinpoint the columns in the 'df' DataFrame exhibiting high correlation
sns.heatmap(df.corr() , annot = True)

# Drop features highly correlated with 'carat'
df.drop(axis = 1 , columns = ['x' , 'y' , 'z'] , inplace = True)
df.head()

# Again build a linear regression model using the remaining features
features = df.columns[:-1]
X = df[features]
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X , Y , test_size = 0.33 , random_state = 42)

# Build linear regression model using the 'sklearn.linear_model' module.
lin_reg = LinearRegression()
lin_reg.fit(X_train , y_train)
# Print the value of the intercept
print("\nConstant".ljust(15, " "), f"{lin_reg.intercept_:.6f}")

# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, lin_reg.coef_)):
  print(f"{i[0]}".ljust(15, " "), f"{i[1]:.6f}")
  


# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
y_train_pred2 = lin_reg.predict(X_train)
y_test_pred2 = lin_reg.predict(X_test)

print(f"Train Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_train, y_train_pred2):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred2):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred2)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred2):.3f}")

print(f"\n\nTest Set\n{'-' * 50}")
print(f"R-squared: {r2_score(y_test, y_test_pred2):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred2):.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred2)):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred2):.3f}")



# Calculate the VIF values for the remaining features using the 'variance_inflation_factor' function.
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Add a constant to feature variables
X_train_sm=sm.add_constant(X_train)
# Create a dataframe that will contain the names of the feature variables and their respective VIFs
VIF = pd.DataFrame()
VIF['features'] = X_train_sm.columns
VIF['vif'] = [variance_inflation_factor(X_train_sm.values , i) for i in range(X_train_sm.values.shape[1])]
VIF['vif'] = round(VIF['vif'] , 2)
VIF = VIF.sort_values(by = 'vif' , ascending = False)
VIF


# Create a histogram for the errors obtained in the predicted values for the train set.
train_errors = y_train - y_train_pred2

plt.title('Histogram for the errors obtained in the predicted values of the train set.')
plt.hist(train_errors , bins = 'sturges' , color = '#3175D9')
plt.axvline(train_errors.mean() , color = '#D93131')



# Create a histogram for the errors obtained in the predicted values for the test set.
test_errors = y_test - y_test_pred2

plt.title('Histogram for the errors obtained in the predicted values of the test set.')
plt.hist(test_errors , bins = 'sturges' , color = '#3175D9')
plt.axvline(test_errors.mean() , color = '#D93131')



# Create a scatter plot between the errors and the dependent variable for the train set.
plt.title('Scatter plot between the errors and the dependent variable for the train set.')
plt.xlabel('Errors')
plt.ylabel('Dependent variable')
plt.scatter(train_errors , y_train , color = '#D93131' , marker = '*')

