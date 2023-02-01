from tensorflow.keras.models import load_model
from data_loader import load_mnist_data
from supervised_models import logit, xgb_model, mlp
from vime_utils import perf_metric



vime_self_encoder = load_model('./save_model/encoder_model.h5')
vime_self_encoder.summary()

# Load data
label_no = 100
label_data_rate = 0.1

x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

# Use subset of labeled data
x_train = x_train[:label_no, :]
y_train = y_train[:label_no, :]  


x_train_hat = vime_self_encoder.predict(x_train)
x_test_hat = vime_self_encoder.predict(x_test)


def supervised_model_training(x_train, y_train, x_test, y_test, model_name, metric):
  # Train supervised model
  # Logistic regression
  if model_name == 'logit':
    y_test_hat = logit(x_train, y_train, x_test)
  # XGBoost
  elif model_name == 'xgboost':
    y_test_hat = xgb_model(x_train, y_train, x_test)      
  # MLP
  elif model_name == 'mlp':    
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100
      
    y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
    
  # Report the performance
  performance = perf_metric(metric, y_test, y_test_hat)    
    
  return performance 


model_name = 'logit'
metric = 'acc'
performance = supervised_model_training(x_train_hat, y_train, x_test_hat, y_test,model_name, metric)
# mlp -> 0.6942
# xgboost -> 0.6097
# logit -> 0.7604