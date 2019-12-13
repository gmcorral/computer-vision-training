import os, json
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon, autograd, ndarray
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

def train(channel_input_dirs, hyperparameters):
    """
    Train a CNN network using the provided data link and hyper-parameters
    :param channel_input_dirs: Data link (s3 link).
    :param hyperparameters: Hyper-parameters for this training task.
    :return: net: The trained Gluon model.
    """
    
    # Set this to CPU or GPU depending on your training instance
    # ctx = mx.cpu()
    ctx = mx.gpu()
    
    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 150)
    epochs = hyperparameters.get('epochs', 10)
    learning_rate = hyperparameters.get('learning_rate', 0.01)
    patience = hyperparameters.get('patience', 5)
    
    training_dir = channel_input_dirs['training']
    X_train = np.load(training_dir + "/training/train_data.npy")
    y_train = np.load(training_dir + "/training/train_label.npy")
    X_val = np.load(training_dir + "/validation/validation_data.npy")
    y_val = np.load(training_dir + "/validation/validation_label.npy")
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    # Create the network. We have 5 classes
    num_outputs = 5
    
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=60, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))
        
    # Initialize parameters
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    # Define loss and trainer.
    softmax_cross_etropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    
    train_metric = mx.metric.Accuracy()
    validation_metric = mx.metric.Accuracy()
    
    highest_validation_acc = 0
    patience_counter = 0

    # Starting the training loop, we will have 50 epochs
    for epoch in range(epochs):
        
        # reset metrics
        train_metric.reset()
        validation_metric.reset()
        
        # training loop (with autograd and trainer steps, etc.)
        cumulative_train_loss = 0
        train_predictions = []
        for i in range(0, X_train.shape[0], batch_size):
            data = nd.array(X_train[i:i + batch_size].astype('float32')).as_in_context(ctx)
            label = nd.array(y_train[i:i + batch_size]).as_in_context(ctx)

            with autograd.record():
                output = net(data)
                train_predictions = train_predictions + np.argmax(output.asnumpy(), axis=1).tolist()
                loss = softmax_cross_etropy_loss(output, label)
                cumulative_train_loss = cumulative_train_loss + nd.sum(loss)
                
            loss.backward()
            trainer.step(data.shape[0])
            train_metric.update([label], [output])
            
        train_loss = cumulative_train_loss/len(X_train)

        # validation loop
        cumulative_valid_loss = 0
        val_predictions = []
        for i in range(0, X_val.shape[0], batch_size):
            data = nd.array(X_val[i:i + batch_size].astype('float32')).as_in_context(ctx)
            label = nd.array(y_val[i:i + batch_size]).as_in_context(ctx)
            output = net(data)
            val_predictions = val_predictions + np.argmax(output.asnumpy(), axis=1).tolist()
            val_loss = softmax_cross_etropy_loss(output, label)
            cumulative_valid_loss = cumulative_valid_loss + nd.sum(val_loss)
            validation_metric.update([label], [output])
            
        valid_loss = cumulative_valid_loss/len(X_val)

        # Print the summary and plot the confusion matrix after each epoch
        
        train_metric_name, train_metric_val = train_metric.get()
        val_metric_name, val_metric_val = validation_metric.get()
        
        # Let's check the highest accuracy and apply the patience logic here.
        if val_metric_val > highest_validation_acc:
            highest_validation_acc = val_metric_val
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
        
        print('[Epoch %d] Training: %s=%f loss: %f' % (epoch, train_metric_name, train_metric_val, train_loss.asnumpy()[0]))
        print('[Epoch %d] Validation: %s=%f loss: %f' % (epoch, val_metric_name, val_metric_val, valid_loss.asnumpy()[0]))
        
        # If we pass the patience threshold, break the loop (stop the training)
        if patience_counter > patience:
            break
            
    return net

def save(net, model_dir):
    net.save_parameters('%s/model.params' % model_dir)
    
# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    
    ctx = mx.cpu()
    
    # Create the network. We have 5 classes
    num_outputs = 5
    
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=60, kernel_size=3, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))
        
    # Initialize parameters
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    # Load the trained model
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    ctx = mx.cpu()
    
    data = np.frombuffer(data, dtype=np.dtype(np.float64))
    data = data.reshape((4, 3, 224, 224))
    nda = nd.array(data.astype('float32')).as_in_context(ctx)
    
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist())
    
    return response_body, output_content_type