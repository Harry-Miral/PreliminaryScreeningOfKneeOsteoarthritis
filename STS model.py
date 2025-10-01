import math

from pandas import read_csv, unique
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import mode
from scipy import interp
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc
from sklearn.metrics import brier_score_loss, precision_score
from sklearn.model_selection import KFold
from sklearn import metrics

from tensorflow import stack
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras.optimizers.legacy import SGD, Adam

import statsmodels.stats.api as sms


import warnings
warnings.filterwarnings("ignore")

def read_data(filepath):
    df = read_csv(filepath, header=None, names=['user-id',
                                               'activity',
                                               'timestamp',
                                               'sex',
                                               'age',
                                               'BMI',
                                               'A',
                                               'B',
                                               'C',
                                               'X',
                                               'Y',
                                               'Z'])
    df['Z'].replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df['Z'] = df['Z'].apply(convert_to_float)
    return df

def convert_to_float(x):
    try:
        return np.float64(x)
    except:
        return np.nan
df = read_data('Dataset/Angel_and_Baseline/Angel_data_STS_order.txt')
df = df.drop(labels=['sex', 'age','BMI'], axis=1)
label_encode = LabelEncoder()
df['activityEncode'] = label_encode.fit_transform(df['activity'].values.ravel())
interpolation_fn = interp1d(df['activityEncode'] ,df['Z'], kind='linear')
null_list = df[df['Z'].isnull()].index.tolist()
for i in null_list:
    y = df['activityEncode'][i]
    value = interpolation_fn(y)
    df['Z']=df['Z'].fillna(value)
    print(value)
df['A'] = (df['A']-df['A'].min())/(df['A'].max()-df['A'].min())
df['B'] = (df['B']-df['B'].min())/(df['B'].max()-df['B'].min())
df['C'] = (df['C']-df['C'].min())/(df['C'].max()-df['C'].min())
df['X'] = (df['X']-df['X'].min())/(df['X'].max()-df['X'].min())
df['Y'] = (df['Y']-df['Y'].min())/(df['Y'].max()-df['Y'].min())
df['Z'] = (df['Z']-df['Z'].min())/(df['Z'].max()-df['Z'].min())

def segments(df, time_steps, step, label_name):
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['X'].values[i:i+time_steps]
        ys = df['Y'].values[i:i+time_steps]
        zs = df['Z'].values[i:i+time_steps]
        aas = df['A'].values[i:i+time_steps]
        bs = df['B'].values[i:i+time_steps]
        cs = df['C'].values[i:i+time_steps]
        label = mode(df[label_name][i:i+time_steps])[0][0]
        segments.append([aas,bs,cs,xs, ys, zs])
        labels.append(label)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    return reshaped_segments, labels
TIME_PERIOD = 80
STEP_DISTANCE = 40
LABEL = 'activityEncode'

df1=df[df['user-id']>70]
df2=df[df['user-id']>140]
df3=df[df['user-id']>210]
df4=df[df['user-id']>280]

a1=df.shape[0]
b2=df1.shape[0]
c3=df2.shape[0]
d4=df3.shape[0]
e5=df4.shape[0]

df_test0 = df.iloc[0:a1-b2,:]
df_train0 = df.iloc[a1-b2:a1,:]
x_train0, y_train0 = segments(df_train0, TIME_PERIOD, STEP_DISTANCE, LABEL)
x_test0, y_test0 = segments(df_test0, TIME_PERIOD, STEP_DISTANCE, LABEL)
time_period, sensors = x_train0.shape[1], x_train0.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_train0 = x_train0.reshape(x_train0.shape[0], input_shape)
x_train0 = x_train0.astype('float32')
y_train0=np.asarray(y_train0).astype('float32').reshape((-1,1))
time_period, sensors = x_test0.shape[1], x_test0.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_test0 = x_test0.reshape(x_test0.shape[0], input_shape)
x_test0 = x_test0.astype('float32')
y_test0=np.asarray(y_test0).astype('float32').reshape((-1,1))

df_test1 = df.iloc[a1-b2:a1-c3,:]
df_train1 = pd.concat([df_test0,df.iloc[a1-c3:a1,]])
x_train1, y_train1 = segments(df_train1, TIME_PERIOD, STEP_DISTANCE, LABEL)
x_test1, y_test1 = segments(df_test1, TIME_PERIOD, STEP_DISTANCE, LABEL)
time_period, sensors = x_train1.shape[1], x_train1.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_train1 = x_train1.reshape(x_train1.shape[0], input_shape)
x_train1 = x_train1.astype('float32')
y_train1=np.asarray(y_train1).astype('float32').reshape((-1,1))
time_period, sensors = x_test1.shape[1], x_test1.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_test1 = x_test1.reshape(x_test1.shape[0], input_shape)
x_test1 = x_test1.astype('float32')
y_test1=np.asarray(y_test1).astype('float32').reshape((-1,1))

df_test2 = df.iloc[a1-c3:a1-d4,:]
df_train2 = pd.concat([df.iloc[0:a1-c3,:],df.iloc[a1-d4:a1,]])
x_train2, y_train2 = segments(df_train2, TIME_PERIOD, STEP_DISTANCE, LABEL)
x_test2, y_test2 = segments(df_test2, TIME_PERIOD, STEP_DISTANCE, LABEL)
time_period, sensors = x_train2.shape[1], x_train2.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_train2 = x_train2.reshape(x_train2.shape[0], input_shape)
x_train2 = x_train2.astype('float32')
y_train2=np.asarray(y_train2).astype('float32').reshape((-1,1))
time_period, sensors = x_test2.shape[1], x_test2.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_test2 = x_test2.reshape(x_test2.shape[0], input_shape)
x_test2 = x_test2.astype('float32')
y_test2=np.asarray(y_test2).astype('float32').reshape((-1,1))

df_test3 = df.iloc[a1-d4:a1-e5,:]
df_train3 = pd.concat([df.iloc[0:a1-d4,:],df.iloc[a1-e5:a1,]])
x_train3, y_train3 = segments(df_train3, TIME_PERIOD, STEP_DISTANCE, LABEL)
x_test3, y_test3 = segments(df_test3, TIME_PERIOD, STEP_DISTANCE, LABEL)
time_period, sensors = x_train3.shape[1], x_train3.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_train3 = x_train3.reshape(x_train3.shape[0], input_shape)
x_train3 = x_train3.astype('float32')
y_train3=np.asarray(y_train3).astype('float32').reshape((-1,1))
time_period, sensors = x_test3.shape[1], x_test3.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_test3 = x_test3.reshape(x_test3.shape[0], input_shape)
x_test3 = x_test3.astype('float32')
y_test3=np.asarray(y_test3).astype('float32').reshape((-1,1))

df_test4 = df.iloc[a1-e5:a1,:]
df_train4 = df.iloc[0:a1-e5,:]
x_train4, y_train4 = segments(df_train4, TIME_PERIOD, STEP_DISTANCE, LABEL)
x_test4, y_test4 = segments(df_test4, TIME_PERIOD, STEP_DISTANCE, LABEL)
time_period, sensors = x_train4.shape[1], x_train4.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_train4 = x_train4.reshape(x_train4.shape[0], input_shape)
x_train4 = x_train4.astype('float32')
y_train4=np.asarray(y_train4).astype('float32').reshape((-1,1))
time_period, sensors = x_test4.shape[1], x_test4.shape[2]
num_classes = label_encode.classes_.size
input_shape = time_period * sensors
x_test4 = x_test4.reshape(x_test4.shape[0], input_shape)
x_test4 = x_test4.astype('float32')
y_test4=np.asarray(y_test4).astype('float32').reshape((-1,1))

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.85,
    patience=15,
    min_lr=1e-7,
    verbose=1
)

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model.add(LSTM(32,return_sequences=True, activation='relu'))
model.add(Reshape((1, 480, 32)))
model.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model.add(Reshape((240, 64)))
model.add(MaxPool1D(pool_size=4, padding='same'))
model.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model.add(Reshape((59, 192)))
model.add(GlobalAveragePooling1D())
model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def step_decay(epoch):
	initial_lrate = 0.0001
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

adam = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric = get_lr_metric(adam)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',lr_metric])

model1 = Sequential()
model1.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model1.add(LSTM(32,return_sequences=True, activation='relu'))
model1.add(Reshape((1, 480, 32)))
model1.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model1.add(Reshape((240, 64)))
model1.add(MaxPool1D(pool_size=4, padding='same'))
model1.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model1.add(Reshape((59, 192)))
model1.add(GlobalAveragePooling1D())
model1.add(BatchNormalization(epsilon=1e-06))
model1.add(Dense(1))
model1.add(Activation('sigmoid'))
adam1 = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric1 = get_lr_metric(adam1)
model1.compile(loss='binary_crossentropy', optimizer=adam1, metrics=['accuracy',lr_metric1])

model2 = Sequential()
model2.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model2.add(LSTM(32,return_sequences=True, activation='relu'))
model2.add(Reshape((1, 480, 32)))
model2.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model2.add(Reshape((240, 64)))
model2.add(MaxPool1D(pool_size=4, padding='same'))
model2.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model2.add(Reshape((59, 192)))
model2.add(GlobalAveragePooling1D())
model2.add(BatchNormalization(epsilon=1e-06))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))
adam2 = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric2 = get_lr_metric(adam2)
model2.compile(loss='binary_crossentropy', optimizer=adam2, metrics=['accuracy',lr_metric2])

model3 = Sequential()
model3.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model3.add(LSTM(32,return_sequences=True, activation='relu'))
model3.add(Reshape((1, 480, 32)))
model3.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model3.add(Reshape((240, 64)))
model3.add(MaxPool1D(pool_size=4, padding='same'))
model3.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model3.add(Reshape((59, 192)))
model3.add(GlobalAveragePooling1D())
model3.add(BatchNormalization(epsilon=1e-06))
model3.add(Dense(1))
model3.add(Activation('sigmoid'))
adam3 = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric3 = get_lr_metric(adam3)
model3.compile(loss='binary_crossentropy', optimizer=adam3, metrics=['accuracy',lr_metric3])

model4 = Sequential()
model4.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model4.add(LSTM(32,return_sequences=True, activation='relu'))
model4.add(Reshape((1, 480, 32)))
model4.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model4.add(Reshape((240, 64)))
model4.add(MaxPool1D(pool_size=4, padding='same'))
model4.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model4.add(Reshape((59, 192)))
model4.add(GlobalAveragePooling1D())
model4.add(BatchNormalization(epsilon=1e-06))
model4.add(Dense(1))
model4.add(Activation('sigmoid'))
adam4 = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric4 = get_lr_metric(adam4)
model4.compile(loss='binary_crossentropy', optimizer=adam4, metrics=['accuracy',lr_metric4])

model5 = Sequential()
model5.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
model5.add(LSTM(32,return_sequences=True, activation='relu'))
model5.add(Reshape((1, 480, 32)))
model5.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model5.add(Reshape((240, 64)))
model5.add(MaxPool1D(pool_size=4, padding='same'))
model5.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model5.add(Reshape((59, 192)))
model5.add(GlobalAveragePooling1D())
model5.add(BatchNormalization(epsilon=1e-06))
model5.add(Dense(1))
model5.add(Activation('sigmoid'))
adam5 = Adam(lr=0.000135, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
lr_metric5 = get_lr_metric(adam5)
model5.compile(loss='binary_crossentropy', optimizer=adam5, metrics=['accuracy',lr_metric5])

scores = []
mean_fpr=np.linspace(0,1,100)
auck=[0,0,0,0,0]
Recalls=[]
Precisions=[]
F1score=[]
specificity=[]
accuracys = []
brier_scores = []

for i in range(5):
    if i==0:
        x_train, y_train = x_train0,y_train0
        x_test, y_test = x_test0,y_test0
        model.fit(x_train, y_train, batch_size=100, epochs=2000, validation_data=(x_test, y_test), callbacks=[early_stopping, lr_reducer])
        score = model.evaluate(x_test, y_test)[1]
        scores.append(score)
        y_pred_proba = model.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tn / (tn + fp))
        accuracys.append(accuracy_score(y_test, y_pred))
        Recalls.append(metrics.recall_score(y_test, y_pred))
        Precisions.append(precision_score(y_test, y_pred))
        F1score.append(metrics.f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_proba))
    elif i ==1:
        x_train, y_train = x_train1,y_train1
        x_test, y_test = x_test1,y_test1
        model1.fit(x_train, y_train, batch_size=100, epochs=2000, validation_data=(x_test, y_test), callbacks=[early_stopping, lr_reducer])
        score = model1.evaluate(x_test, y_test)[1]
        scores.append(score)
        y_pred_proba = model1.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tn / (tn + fp))
        accuracys.append(accuracy_score(y_test, y_pred))
        Recalls.append(metrics.recall_score(y_test, y_pred))
        Precisions.append(precision_score(y_test, y_pred))
        F1score.append(metrics.f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_proba))
    elif i ==2:
        x_train, y_train = x_train2,y_train2
        x_test, y_test = x_test2,y_test2
        model2.fit(x_train, y_train, batch_size=100, epochs=2000, validation_data=(x_test, y_test), callbacks=[early_stopping, lr_reducer])
        score = model2.evaluate(x_test, y_test)[1]
        scores.append(score)
        y_pred_proba = model2.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tn / (tn + fp))
        accuracys.append(accuracy_score(y_test, y_pred))
        Recalls.append(metrics.recall_score(y_test, y_pred))
        Precisions.append(precision_score(y_test, y_pred))
        F1score.append(metrics.f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_proba))
    elif i ==3:
        x_train, y_train = x_train3,y_train3
        x_test, y_test = x_test3,y_test3
        model3.fit(x_train, y_train, batch_size=100, epochs=2000, validation_data=(x_test, y_test), callbacks=[early_stopping, lr_reducer])
        score = model3.evaluate(x_test, y_test)[1]
        scores.append(score)
        y_pred_proba = model3.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tn / (tn + fp))
        accuracys.append(accuracy_score(y_test, y_pred))
        Recalls.append(metrics.recall_score(y_test, y_pred))
        Precisions.append(precision_score(y_test, y_pred))
        F1score.append(metrics.f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_proba))
    else:
        x_train, y_train = x_train4,y_train4
        x_test, y_test = x_test4,y_test4
        model4.fit(x_train, y_train, batch_size=100, epochs=2000, validation_data=(x_test, y_test), callbacks=[early_stopping, lr_reducer])
        score = model4.evaluate(x_test, y_test)[1]
        scores.append(score)
        y_pred_proba = model4.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tn / (tn + fp))
        accuracys.append(accuracy_score(y_test, y_pred))
        Recalls.append(metrics.recall_score(y_test, y_pred))
        Precisions.append(precision_score(y_test, y_pred))
        F1score.append(metrics.f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_proba))
    auck[i] = auc(fpr, tpr)

scores=np.array(scores)
accuracys=np.array(accuracys)
Recalls=np.array(Recalls)
Precisions=np.array(Precisions)
specificity=np.array(specificity)
F1score=np.array(F1score)
auck=np.array(auck)
brier_scores=np.array(brier_scores)

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    return lower, upper

print("\n========== Mean Metrics and 95% Confidence Intervals (Bootstrap) ==========")
print(f"Accuracy: {np.mean(accuracys):.3f}, 95% CI: {bootstrap_ci(accuracys)}")
print(f"Precision: {np.mean(Precisions):.3f}, 95% CI: {bootstrap_ci(Precisions)}")
print(f"Recall: {np.mean(Recalls):.3f}, 95% CI: {bootstrap_ci(Recalls)}")
print(f"Specificity: {np.mean(specificity):.3f}, 95% CI: {bootstrap_ci(specificity)}")
print(f"F1-Score: {np.mean(F1score):.3f}, 95% CI: {bootstrap_ci(F1score)}")
print(f"AUC: {np.mean(auck):.3f}, 95% CI: {bootstrap_ci(auck)}")
print(f"Brier Score: {np.mean(brier_scores):.3f}, 95% CI: {bootstrap_ci(brier_scores)}")