"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

# from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *

from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout
from keras.layers.pooling import GlobalAveragePooling1D

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

#

# TODO: advanced activations - 'leakyrelu', 'prelu', 'elu', 'thresholdedrelu', 'srelu' 


max_layers = 2
max_layer_size = 100
max_conv_layers = 10
J = 5
I = 9

space = {
    'scaler': hp.choice('s',
                        (None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler')),
    'j_neuron': hp.quniform('J', 1, J, 1),
    'i_gate': hp.quniform('I', 1, I, 1),
    'init': hp.choice('init', ('uniform', 'normal', 'glorot_uniform',
                            'glorot_normal', 'he_uniform', 'he_normal')),
    'batch_size': hp.choice('batch_size', (16, 32, 64, 128, 256)),
    'shuffle': hp.choice('shuffle', (False, True)),
    'loss': hp.choice('l', ('mean_absolute_error', 'mean_squared_error')),
    'optimizer': hp.choice('o', ('rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax'))
}


# for each hidden layer, we choose size, activation and extras individually
for i in range(1, max_layers + 1):
    space['layer_{}_size'.format(i)] = hp.quniform('ls{}'.format(i),
                                                   2, max_layer_size, 1)
    space['layer_{}_bn'.format(i)] = hp.choice('bn{}'.format(i), [True, False])
    space['layer_{}_activation'.format(i)] = hp.choice('a{}'.format(i),
                                                       ('relu', 'sigmoid', 'tanh'))
    space['layer_{}_extras'.format(i)] = hp.choice('e{}'.format(i), (
        {'name': 'dropout', 'rate': hp.uniform('d{}'.format(i), 0.1, 0.5)},
        {'name': None}))


def get_params():
    params = sample(space)
    return handle_integers(params)


def try_params(n_iterations, params):
    print("iterations:", n_iterations)
    #print_params(params)

    y_train = data['y_train']
    y_test = data['y_test']

    if params['scaler']:
        scaler = eval("{}()".format(params['scaler']))
        x_train_ = scaler.fit_transform(data['x_train'].astype(float))
        x_test_ = scaler.transform(data['x_test'].astype(float))
    else:
        x_train_ = data['x_train']
        x_test_ = data['x_test']

    input_dim = x_train_.shape[1]

    model = Sequential()
    w = []
    b = []
    m = 0.1

    for j in range(0, int(params['j_neuron'])):
        #TODO: keras customized layer? b[j]-m*np.exp(w[j])

    for i in range(1, int(params['n_layers']) + 1):
        if i == 1:
            model.add(Dense(params['layer_{}_size'.format(i)], kernel_initializer=params['init'],
                            input_shape=x_train_.shape[1:]))
        else:
            model.add(Dense(params['layer_{}_size'.format(i)], kernel_initializer=params['init']))

        if params['layer_{}_bn'.format(i)]:
            model.add(BatchNormalization())
        model.add(Activation(params['layer_{}_activation'.format(i)]))
        extras = 'layer_{}_extras'.format(i)
        if not params['layer_{}_bn'.format(i)]:
            model.add(BatchNormalization())
        if params[extras]['name'] == 'dropout':
            model.add(Dropout(params[extras]['rate']))

    model.add(Dense(1, kernel_initializer=params['init'], activation='linear'))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])

    # print model.summary()

    #

    validation_data = (x_test_, y_test)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    history = model.fit(x_train_, y_train,
                        epochs=int(round(n_iterations)),
                        batch_size=params['batch_size'],
                        shuffle=params['shuffle'],
                        validation_data=validation_data,
                        callbacks=[early_stopping])

    #

    p = model.predict(x_train_, batch_size=params['batch_size'])

    mse = MSE(y_train, p)
    rmse = sqrt(mse)
    mae = MAE(y_train, p)

    print("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))

    #

    p = model.predict(x_test_, batch_size=params['batch_size'])

    mse = MSE(y_test, p)
    rmse = sqrt(mse)
    mae = MAE(y_test, p)

    print("# testing  | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))

    return {'loss': rmse, 'rmse': rmse, 'mae': mae, 'early_stop': model.stop_training}
