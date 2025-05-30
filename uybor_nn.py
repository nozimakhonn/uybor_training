import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

df_houses = pd.read_csv('houses_new.csv')
df_flats = pd.read_csv('flats_new.csv')
df_houses['home_type'] = 'house'
df_flats['home_type'] = 'flat'
df_houses.info()
print(df_houses.head())
# df_flats.info()
# df_houses.info()

numerical_features = ['rooms', 'floors', 'facilities', 'home_area(acres²)','negotiations', 'living_squares', 'total_floors']

categorical_features = ['renovation', 'materials', 'city', 'street', 'district']
target = 'prices'

# df_houses[target] = pd.to_numeric(df_houses[target], errors='coerce')  # Convert non-numeric values to NaN
# df_houses[target].fillna(df_houses[target].median(), inplace=True)  # Replace NaNs with median
# y = np.log1p(df_houses[target])
def preprocessing(df):
    sc = StandardScaler()
    oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', sc)
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', oe)
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features),
        ]
    )
    sparse_threshold=0
    features = numerical_features + categorical_features
    x = df[features]
    y = df[target]
    # y = pd.to_numeric(y, errors='coerce')
    # y.fillna(y.median(), inplace=True)
    x_processed = preprocessor.fit_transform(x)
    y_processed = np.log1p(y)

    return x_processed, y_processed, preprocessor, features

class HousePrices(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        x = torch.relu(self.layer4(x))
        x = self.dropout(x)
        x = torch.relu(self.layer5(x))
        return x

x_houses_pro, y_houses_pro, preprocessor_houses, features_houses = preprocessing(df_houses)
x_train_houses, x_test_houses, y_train_houses, y_test_houses = train_test_split(x_houses_pro, y_houses_pro,
                                                                                test_size=25)
x_train_tensor = torch.FloatTensor(x_train_houses)
x_test_tensor = torch.FloatTensor(x_test_houses)
y_train_tensor = torch.FloatTensor(y_train_houses.values).reshape(-1,1)
y_test_tensor = torch.FloatTensor(y_test_houses.values).reshape(-1,1)

def train_model(input_size, x_train_tensor, y_train_tensor):
    model = HousePrices(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 1000
    batch_size = 25
    for j in range(num_epoch):
        model.train()
        epoch_loss = 0
        for i in range(0, len(x_train_tensor), batch_size):
            batch_x = x_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            if os.path.exists('house_prices_model.pth'):
                print("Loading trained model...")
                model.load_state_dict(torch.load('house_price_model.pth'))
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(x_train_tensor / batch_size)
        print(f"Epoch: {j+1} Avg Loss: {avg_train_loss}")
    torch.save(model.state_dict(), 'houses_adv_model.pth')
    return model

input_size = x_test_tensor.shape[1]
trained_model = train_model(input_size, x_train_tensor, y_train_tensor)

def evaluate_model(model, x_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_mse = mean_squared_error(y_test_tensor, test_outputs.numpy())
        test_mse = np.sqrt(test_mse)
    print('Test MSE:', test_mse)
    return test_outputs

test_outputs = evaluate_model(trained_model, x_test_tensor, y_test_tensor)
new_house = pd.DataFrame(
        {
            'rooms': [7],
            'floors': [1],
            'facilities': [1],
            'home_area(acres²)': [6],
            'negotiations': [1],
            'living_squares': [400],
            'total_floors': [1],
            'renovation': ['Medium'],
            'materials': ['Brick'],
            'city': ['Tashkent city'],
            'street': ['Nurhon street'],
            'district': ['Chilanzar district'],
            'home_type': ['house']
        }
    )
new_house_pro = preprocessor_houses.transform(new_house[numerical_features+categorical_features])
new_house_tensor = torch.FloatTensor(new_house_pro)
trained_model.eval()
with torch.no_grad():
    new_house_pred = trained_model(new_house_tensor)
    predicted_price = np.expm1(new_house_pred.numpy()[0][0])
print(f"Predicted house price: {predicted_price:,.2f}")

data_verify = pd.DataFrame(y_test_tensor.numpy(), columns=['Test'])
data_predicted = pd.DataFrame(test_outputs.numpy(), columns=['Predicted'])

output_data = pd.concat([data_verify, data_predicted], axis=1)
output_data['Difference'] = output_data['Test'] - output_data['Predicted']
print(output_data.head())

output_data['Test'] = np.expm1(output_data['Test'])
output_data['Predicted'] = np.expm1(output_data['Predicted'])
output_data['Difference'] = output_data['Test'] - output_data['Predicted']
print(output_data)
df_houses.info()

x_flats_pro, y_flats_pro, preprocessor_flats, features_flats = preprocessing(df_flats)
x_train_flats, x_test_flats, y_train_flats, y_test_flats = train_test_split(x_flats_pro, y_flats_pro,
                                                                                test_size=25)
x_train_tensor_fl = torch.FloatTensor(x_train_flats)
x_test_tensor_fl = torch.FloatTensor(x_test_flats)
y_train_tensor_fl = torch.FloatTensor(y_train_flats.values).reshape(-1,1)
y_test_tensor_fl = torch.FloatTensor(y_test_flats.values).reshape(-1,1)

input_size_fl = x_train_tensor_fl.shape[1]
trained_model_fl = train_model(input_size_fl, x_train_tensor_fl, y_train_tensor_fl)
test_outputs_fl = evaluate_model(trained_model_fl, x_test_tensor_fl, y_test_tensor_fl)
new_flat = pd.DataFrame(
        {
            'rooms': [7],
            'floors': [1],
            'facilities': [1],
            'home_area(acres²)': [6],
            'negotiations': [1],
            'living_squares': [400],
            'total_floors': [1],
            'renovation': ['Medium'],
            'materials': ['Brick'],
            'city': ['Tashkent city'],
            'street': ['Nurhon street'],
            'district': ['Chilanzar district'],
            'home_type': ['flat']
        }
    )
new_flat_pro = preprocessor_flats.transform(new_flat[numerical_features+categorical_features])
new_flat_tensor = torch.FloatTensor(new_flat_pro)
trained_model.eval()
with torch.no_grad():
    new_flat_pred = trained_model_fl(new_flat_tensor)
    predicted_price = np.expm1(new_flat_pred.numpy()[0][0])
print(f"Predicted flat price: {predicted_price:,.2f}")

data_verify = pd.DataFrame(y_test_tensor.numpy(), columns=['Test'])
data_predicted = pd.DataFrame(test_outputs.numpy(), columns=['Predicted'])


output_data = pd.concat([data_verify, data_predicted], axis=1)
output_data['Difference'] = output_data['Test'] - output_data['Predicted']
print(output_data.head())

output_data['Test'] = np.expm1(output_data['Test'])
output_data['Predicted'] = np.expm1(output_data['Predicted'])
output_data['Difference'] = output_data['Test'] - output_data['Predicted']
print(output_data)
df_flats.info()
