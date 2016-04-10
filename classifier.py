import theano
import theano.tensor as T
import numpy
rng = numpy.random

floatX = theano.config.floatX


class Classifier(object):
    def __init__(self, n_features):
        hidden_layer_size = 5
        l2_regularisation = 0.001

        input_vector = T.fvector('input_vector')
        target_value = T.fscalar('target_value')
        learningrate = T.fscalar('learningrate')

        W_hidden_vals = numpy.asarray(rng.normal(loc=0.0,
                                                 scale=0.1,
                                                 size=(n_features, hidden_layer_size)),
                                      dtype=floatX)
        W_hidden = theano.shared(W_hidden_vals, 'W_hidden')

        hidden = T.dot(input_vector, W_hidden)
        hidden = T.nnet.sigmoid(hidden)

        W_output_vals = numpy.asarray(rng.normal(loc=0.0,
                                                 scale=0.1,
                                                 size=(hidden_layer_size, 1)),
                                      dtype=floatX)
        W_output = theano.shared(W_output_vals, 'W_output')

        predicted_value = T.dot(hidden, W_output)
        predicted_value = T.nnet.sigmoid(predicted_value)

        cost = T.sqr(predicted_value - target_value).sum()
        cost += l2_regularisation * (T.sqr(W_hidden).sum() + T.sqr(W_output).sum())

        params = [W_hidden, W_output]
        gradients = T.grad(cost, params)
        updates = [(p, p - (learningrate * g)) for p, g in zip(params, gradients)]

        self.train = theano.function(inputs=[input_vector, target_value, learningrate],
                                     outputs=[cost, predicted_value],
                                     updates=updates,
                                     allow_input_downcast=True)

        self.test = theano.function(inputs=[input_vector, target_value],
                                    outputs=[cost, predicted_value],
                                    allow_input_downcast=True)


def create_random_dataset(num_of_classes, num_of_vectors, num_of_features, feature_min_value, features_max_value):
    dataset = []
    for i in range(num_of_vectors):
        classify_as = rng.randint(1, num_of_classes)
        features_vector = rng.uniform(feature_min_value, features_max_value, num_of_features)
        dataset.append((classify_as, features_vector))
    return dataset


def create_classes_ranges(num_of_classes, min_value, max_value):
    weights_ranges = [min_value]
    diff = (max_value - min_value) / num_of_classes
    previous = min_value
    for i in range(1, num_of_classes):
        previous += diff
        weights_ranges.append(previous)
    weights_ranges.append(max_value)
    return weights_ranges


if __name__ == "__main__":
    learningrate = 0.1
    epochs = 25

    num_classes = 4
    num_features = 4
    data_train_size = 100
    data_test_size = 20

    data_train = create_random_dataset(num_classes, data_train_size, num_features, -1.0, 1.0)
    data_test = create_random_dataset(num_classes, data_test_size, num_features, -1.0, 1.0)

    weights_values = create_classes_ranges(num_classes, 0.0, 1.0)
    print(weights_values)

    classifier = Classifier(num_features)

    print("Neural network test...")
    print("Number of classes: ", num_classes)
    print("Number of features: ", num_features)
    print("Epochs: ", epochs)
    print("Training data:")
    print(data_train)
    print("Testing data:")
    print(data_test)

    for epoch in range(epochs):
        cost_sum = 0.0
        correct = 0
        for label, vector in data_train:
            cost, predicted_value = classifier.train(vector, label, learningrate)
            cost_sum += cost
            range_index = int(label)
            if (predicted_value >= weights_values[range_index]) and (predicted_value < weights_values[range_index + 1]):
                correct += 1
        print("Epoch: " + str(epoch) + " Cost: " + str(cost_sum), ", Accuracy: " + str(float(correct) / len(data_train)))

    cost_sum = 0.0
    correct = 0
    index = 0
    for label, vector in data_test:
        index += 1
        cost, predicted_value = classifier.test(vector, label)
        cost_sum += cost
        range_index = int(label)
        if (predicted_value >= weights_values[range_index]) and (predicted_value < weights_values[range_index + 1]):
            correct += 1
        print("Test_cost: " + str(cost_sum) + ", Test_accuracy: " + str(float(correct) / index))
    print("Number of correct guesses: ", correct, " out of ", data_test_size)