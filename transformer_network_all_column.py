#install yfinance
!pip install yfinance
#install yahoofinancials
!pip install yahoofinancials

#import libraries
import numpy as np
import pandas as pd
import os, datetime
from yahoofinancials import YahooFinancials
import yfinance as yf
import tensorflow as tf
from scipy import stats
import seaborn as sns
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# Download data: Initialize YahooFinancials with the S&P 500 index symbol (^GSPC) for financial data retrieval.
yahoo_financials = YahooFinancials('^GSPC')

# Fetch historical daily price data for ^GSPC from May 5, 1982, to May 5, 2023.
data = yahoo_financials.get_historical_price_data(start_date='1982-05-05',
                                                  end_date='2023-05-05',
                                                  time_interval='daily')

"""
The preprocessing steps in the provided code can be summarized as follows:

1. **Fetching and Formatting Data:**
   - Historical price data for the S&P 500 index is retrieved using the `YahooFinancials` library.
   - The fetched data is converted into a pandas DataFrame, focusing on the 'prices' key.

2. **Data Cleaning:**
   - The 'date' column is dropped, and 'formatted_date' is set as the new index.
   - Unnecessary columns are removed or rearranged to ensure the data contains only relevant features.

3. **Data Exploration:**
   - Basic descriptive statistics are generated to understand the data better.
   - The DataFrame's shape is checked, and null values are identified and addressed.

4. **Data Transformation:**
   - The 'Volume' column values of 0 are replaced with the forward fill method to handle missing or zero values.
   - The DataFrame is sorted by date.
   - Columns are renamed for clarity and consistency.

5. **Feature Engineering:**
   - Percentage changes (returns) in the 'Adj Close' column are calculated to represent daily returns.
   - A 5-day rolling volatility measure is calculated to understand the price fluctuations over time.

6. **Normalization:**
   - Price and volume columns are normalized to a 0-1 range using the min-max normalization method, based on historical data to avoid lookahead bias.

7. **Data Splitting:**
   - The dataset is split into training, validation, and test sets based on time, ensuring that the model can be evaluated on unseen data.

8. **Handling NaN Values:**
   - Rows containing NaN values are dropped to ensure the model trains on clean data.

9. **Preparing Time Series Data:**
   - The dataset is structured into sequences and targets to fit the model's input requirements. This involves creating subsequences of a specified length (`seq_len`) as input features and the subsequent value as the target.

10. **Visualization:**
    - Various visualizations are created to explore the adjusted close prices, volume over time, and the distribution of daily returns and log returns. Additionally, the volatility trends and the segmentation of the dataset into training, validation, and test sets are visualized to aid in understanding the data's characteristics and the model's performance potential.

"""

# Convert the fetched price data into a pandas DataFrame, focusing on the 'prices' key.
df = pd.DataFrame(data['^GSPC']['prices'])

# Drop the 'date' column and set 'formatted_date' as the new index for the DataFrame.
df = df.drop('date', axis=1).set_index('formatted_date')

# Rearranging the DataFrame columns as specified
df = df[['high', 'low', 'open', 'close', 'adjclose' , 'volume']]

# Display the last five entries in the DataFrame to check the most recent data.
df.tail()

# Descriptive statistics for the DataFrame
df.describe()

#shape of data
df.shape

#check null values
df.isnull().sum()

# Renaming axis and columns as specified
df = df.rename_axis('formatted_date').reset_index()
df = df.rename(columns={"formatted_date": "Date", "high": "High", "low":"Low", "open":"Open", "close":"Close", "volume":"Volume", "adjclose" : "Adj Close"})

#hyperparameters
batch_size = 32
seq_len = 12

d_k = 256
d_v = 256
n_heads = 8
ff_dim = 512

# Applying modifications to the DataFrame as specified
# Replacing 0 in the 'Volume' column with the forward fill method to handle missing or zero values
df['Volume'].replace(to_replace=0, method='ffill', inplace=True)

# Sorting the DataFrame by Date
df.sort_values('Date', inplace=True)

#del df['adjclose']

# Drop all rows with NaN values
df.dropna(how='any', axis=0, inplace=True)
df.head(), df.shape

#Visualizing S&P 500 Adjusted Close Prices over time
fig = plt.figure(figsize=(15,10))
st = fig.suptitle("S&P 500 Adj Close Price and Volume", fontsize=20)
st.set_y(0.92)

ax1 = fig.add_subplot(211)
ax1.plot(df['Adj Close'], label='S&P 500 Adj Close Price')
ax1.set_xticks(range(0, df.shape[0], 1464))
ax1.set_xticklabels(df['Date'].loc[::1464])
ax1.set_ylabel('Adj Close Price', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)

#Visualizing S&P 500 volume over time
ax2 = fig.add_subplot(212)
ax2.plot(df['Volume'], label='S&P 500 Volume')
ax2.set_xticks(range(0, df.shape[0], 1464))
ax2.set_xticklabels(df['Date'].loc[::1464])
ax2.set_ylabel('Volume', fontsize=18)
ax2.legend(loc="upper left", fontsize=12)

#Analyzing s&p 500 returns: normal distribution of daily returns and log returns
returns = 100 * df['Adj Close'].pct_change().dropna()
log_returns = np.log(df['Adj Close']/df['Adj Close'].shift(1)).dropna()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
sns.distplot(returns, norm_hist=True, fit=stats.norm,
             bins=50, ax=ax1)
ax1.set_title('Returns')

sns.distplot(log_returns, norm_hist=True, fit=stats.norm,
             bins=50, ax=ax2)
ax2.set_title('Log Returns')

fig.show();

## Calculating percentage change in the 'Adj Close' column to represent daily returns
df['Adj Close'] = df['Adj Close'].pct_change()

#visualizing daily returns of s&p 500 over time
df.plot(x = 'Date', y = 'Adj Close')
plt.title("Returns")
plt.show()

#volatility of a time series using the standard deviation(Std)
def get_volatility(ts, window=None):
    if window:
        ma = ts.rolling(window).mean()
    else:
        ma = ts.expanding().mean()
    output = ((ts - ma)**2)**.5

    return output

#"Calculating 5-Day Rolling Volatility of S&P 500 Returns"
df['Volatility'] = df['Adj Close'].rolling(5).std()

#S&P 500: Visualizing 5-Day Rolling Volatility Trends
df.plot(x = 'Date', y = 'Volatility')
plt.title("Volatility of S&P 500 ")
plt.show()

#Calculate percentage change - Create arithmetic returns column

df['Open'] = df['Open'].pct_change()
df['High'] = df['High'].pct_change()
df['Low'] = df['Low'].pct_change()
df['Close'] = df['Close'].pct_change()
df['Volume'] = df['Volume'].pct_change()

# Drop all rows with NaN values
df.dropna(how='any', axis=0, inplace=True)

#Create indexes to split dataset
times = sorted(df.index.values)
last_10pct = sorted(df.index.values)[-int(0.15*len(times))] # Last 10% of series
last_20pct = sorted(df.index.values)[-int(0.3*len(times))] # Last 20% of series

#Normalize price columns
min_return = min(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].min(axis=0))
max_return = max(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].max(axis=0))

# Min-max normalize price columns (0-1 range)
df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
df['High'] = (df['High'] - min_return) / (max_return - min_return)
df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

#Normalize volume feature
min_volume = df[(df.index < last_20pct)]['Volume'].min(axis=0)
max_volume = df[(df.index < last_20pct)]['Volume'].max(axis=0)

# Min-max normalize volume feature (0-1 range)
df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)
df.head(5)

#Normalize Adj close feature
min_adj = df[(df.index < last_20pct)]['Adj Close'].min(axis=0)
max_adj = df[(df.index < last_20pct)]['Adj Close'].max(axis=0)

# Min-max normalize Adj close feature (0-1 range)
df['Adj Close'] = (df['Adj Close'] - min_adj) / (max_adj - min_adj)

#Split data to training, validation and test sets

df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
df_test = df[(df.index >= last_10pct)]

# Remove columns
df_train.drop(columns=['Date', 'High','Low','Open', 'Close', 'Adj Close', 'Volume'], inplace=True)
df_val.drop(columns=['Date', 'High','Low','Open', 'Close', 'Adj Close', 'Volume'], inplace=True)
df_test.drop(columns=['Date', 'High','Low','Open', 'Close', 'Adj Close', 'Volume'], inplace=True)

# Convert pandas columns into arrays
train_data = df_train.values
val_data = df_val.values
test_data = df_test.values
print('Training data shape: {}'.format(train_data.shape))
print('Validation data shape: {}'.format(val_data.shape))
print('Test data shape: {}'.format(test_data.shape))

#Visualizing Training, Validation, and Test data segmentation for volatility analysis
fig = plt.figure(figsize=(15,12))
st = fig.suptitle("Data Separation", fontsize=20)
st.set_y(0.95)

ax1 = fig.add_subplot(211)
ax1.plot(np.arange(train_data.shape[0]), df_train['Volatility'], label='Training data')

ax1.plot(np.arange(train_data.shape[0],
                   train_data.shape[0]+val_data.shape[0]), df_val['Volatility'], label='Validation data')

ax1.plot(np.arange(train_data.shape[0]+val_data.shape[0],
                   train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['Volatility'], label='Test data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Closing Returns')
ax1.set_title("Volatility", fontsize=18)
ax1.legend(loc="best", fontsize=12)

#preparing time series data for training set: sequences and targets
X_train, y_train = [], []
for i in range(seq_len, len(train_data)):
  X_train.append(train_data[i-seq_len:i])
  y_train.append(train_data[:, 0][i])
X_train, y_train = np.array(X_train), np.array(y_train)

#preparing time series data for validation set: sequences and targets
X_val, y_val = [], []
for i in range(seq_len, len(val_data)):
    X_val.append(val_data[i-seq_len:i])
    y_val.append(val_data[:, 0][i])
X_val, y_val = np.array(X_val), np.array(y_val)

#preparing time series data for test set: sequences and targets
X_test, y_test = [], []
for i in range(seq_len, len(test_data)):
    X_test.append(test_data[i-seq_len:i])
    y_test.append(test_data[:, 0][i])
X_test, y_test = np.array(X_test), np.array(y_test)

print('Training set shape', X_train.shape, y_train.shape)
print('Validation set shape', X_val.shape, y_val.shape)
print('Testing set shape' ,X_test.shape, y_test.shape)

"""
***** Implementation of Transformer Network *****
Incorporating Time2Vec as positional encoding significantly boosts the performance of time series forecasting models, particularly for financial market data like the S&P 500 index.
Time2Vec's unique ability to capture both periodic and non-periodic patterns enriches the model's comprehension of temporal dynamics, facilitating more precise predictions by
leveraging historical data insights.
"""

#implementing Time2Vector: enhancing time series models with periodic and non-periodic features
class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    #Initialize weights and biases with shape (batch, seq_len)
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    #Calculate periodic and non-periodic time features
    x = tf.math.reduce_mean(x[:,:,:1], axis=-1)
    time_linear = self.weights_linear * x + self.bias_linear # non-periodic time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)

    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config

#implementing Single-head Attention and Multi-head Attention 

#Building a Single-head Attention Layer
class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k,
                       input_shape=input_shape,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')

    self.key = Dense(self.d_k,
                     input_shape=input_shape,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform')

    self.value = Dense(self.d_v,
                       input_shape=input_shape,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)

    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out

#Building a Multi head Attention Layer
class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

    self.linear = Dense(input_shape[0][-1],
                        input_shape=input_shape,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear

#Transformer Network encoder layer: integrating attention and CNN and LSTM layers as feed forward layers
class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    # Initialize multi-head attention with specified dimensions and number of heads
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)

    # Apply dropout to the attention output to prevent overfitting
    self.attn_dropout = Dropout(self.dropout_rate)

    # Normalize the attention output to stabilize the learning process
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    # First 1D convolution layer
    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')

    # Second 1D convolution layer
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)

    # First LSTM layer
    self.ff_lstm_1 = LSTM(units=self.ff_dim, activation='tanh', return_sequences=True)

    # Second LSTM layer
    self.ff_lstm_2 = LSTM(units=input_shape[0][-1], activation='tanh', return_sequences=True)

    # Apply dropout after the LSTM layers to reduce overfitting
    self.ff_dropout = Dropout(self.dropout_rate)

    # Normalization
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)


    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_lstm_1(attn_layer)
    ff_layer = self.ff_lstm_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config

def create_model():
  # Initialize time embedding and transformer encoder layers with specified configurations
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  # Define the input sequence structure
  in_seq = Input(shape=(seq_len, 1))
  # Apply time embedding to the input sequence
  x = time_embedding(in_seq)
  # Concatenate the original input sequence with its time embedding
  x = Concatenate(axis=-1)([in_seq, x])

  # ***Encoder block ***

  # Pass the concatenated input through three transformer encoder layers
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))

  # ***Decoder block ***

  # apply global average pooling to summarize the transformer outputs
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
  # Add dropout for regularization
  x = Dropout(0.1)(x)
  # Apply a dense layer with relu activation for further processing
  x = Dense(64, activation='relu')(x)
  # Another dropout for regularization
  x = Dropout(0.1)(x)
  # Output layer with linear activation to predict the continuous value
  out = Dense(1, activation='linear')(x)

  # Construct and compile the model with mean squared error loss and adam optimizer, including MAE and MAPE metrics for evaluation
  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
  return model


# Create and summarize the model
model = create_model()
model.summary()

# Define a ModelCheckpoint callback to save the model during training
callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5',
                                              monitor='val_loss',
                                              save_best_only=True, verbose=1)
# Train the model with the given dataset
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=50,
                    callbacks=[callback],
                    validation_data=(X_val, y_val))

#Calculate predication for training, validation and test data
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

#Evaluation
#Print evaluation metrics for all datasets
train_eval = model.evaluate(X_train, y_train, verbose=0)
val_eval = model.evaluate(X_val, y_val, verbose=0)
test_eval = model.evaluate(X_test, y_test, verbose=0)
print(' ')
print('Evaluation metrics')
print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

# Visualizing the Model's Performance
fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig.add_subplot(311)
ax11.plot(train_data[:, 0], label='S&P 500 Volatility')
ax11.plot(np.arange(seq_len, train_pred.shape[0]+seq_len), train_pred, linewidth=3, label='Predicted S&P 500  Volatility')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Date')
ax11.set_ylabel('S&P 500  Volatility')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig.add_subplot(312)
ax21.plot(val_data[:, 0], label='S&P 500  Volatility')
ax21.plot(np.arange(seq_len, val_pred.shape[0]+seq_len), val_pred, linewidth=3, label='Predicted S&P 500  Volatility')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Date')
ax21.set_ylabel('S&P 500  Volatility')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig.add_subplot(313)
ax31.plot(test_data[:, 0], label='S&P 500  Volatility')
ax31.plot(np.arange(seq_len, test_pred.shape[0]+seq_len), test_pred, linewidth=3, label='Predicted S&P 500  Volatility')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Date')
ax31.set_ylabel('S&P 500  Volatility')
ax31.legend(loc="best", fontsize=12)

# Display Model Performance Metrics
fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model Metrics", fontsize=22)
st.set_y(0.92)

#Plot Mean Squared Error (MSE)
ax1 = fig.add_subplot(311)
ax1.plot(history.history['loss'], label='Training loss (MSE)')
ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
ax1.set_title("Model loss", fontsize=18)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend(loc="best", fontsize=12)

#Plot Mean Absolute Error (MAE)
ax2 = fig.add_subplot(312)
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean average error (MAE)')
ax2.legend(loc="best", fontsize=12)

#Plot Mean Absolute Percentage Error (MAPE)
ax3 = fig.add_subplot(313)
ax3.plot(history.history['mape'], label='Training MAPE')
ax3.plot(history.history['val_mape'], label='Validation MAPE')
ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Mean average percentage error (MAPE)')
ax3.legend(loc="best", fontsize=12)
