#specify architecture
#compile 
#fit 
#save model
#reload model 
#predict 
#evaluate 
import keras 
from keras.layers import Dense 
from keras.models import Sequential 
from keras.models import load_model
from keras.callbacks import EarlyStopping
import numpy as np 
import pandas as pd 

#import data 
df = pd.read_csv('hourly_wages.csv')
print(df.shape)

target_df = df['wage_per_hour']
feature_df = df.drop(columns = ['wage_per_hour'])

print(target_df.shape)
print(feature_df.shape)

#corvert numpy matrix 
predtictor = feature_df.values
target = target_df.values
#get the number of columns 
n_cols = predtictor.shape[1]
#print(len(predtictor))

#specify model 
model = Sequential()

#specify layer 
#1st layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

#2nd layer 
model.add(Dense(32, activation='relu'))

#output layer 
model.add(Dense(1))

#compile 
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#stop when see model not improving 
early_stopping_monitor = EarlyStopping(patience=2)

#fit
model.fit(predtictor, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])

#save
model.save('hourly_wages.h5')
model.summary()

#reload 
my_model = load_model('hourly_wages.h5')
# predict  


