import numpy as np
from sklearn.dummy import DummyClassifier
from collections import Counter
def dummy_classifier(y_train, n_test, strategy, constant=None):
    if strategy == "uniform":
        dictionary = Counter(y_train)
        k , keys = len(dictionary) , sorted(dictionary.keys())
        x_test = []
        for i in range(n_test):
            x_test.append(keys[i%k])
        return x_test
    elif strategy == "stratified":
        dictionary = Counter(y_train)
        keys = sorted(dictionary.keys())
        exact = {}
        for key in keys:
            frequency = dictionary[key] / len(y_train)
            exact[key] = n_test * frequency
        counts = {}
        for key in keys:
            counts[key] = int(np.floor(exact[key]))
        remaining = n_test - sum(counts.values())
        fractional = {}
        for key in keys:
            fractional[key] = exact[key] - counts[key]
        keys_by_fraction = sorted(
            keys,
            key=lambda key: fractional[key],
            reverse=True
        )
        for key in keys_by_fraction[:remaining]:
            counts[key] += 1
        output = []
        for key in keys:
            output.extend([key] * counts[key])
        return output

    X_train = np.zeros((len(y_train), 1))
    X_test = np.zeros((n_test, 1))

    classifier = DummyClassifier(
        strategy=strategy,
        constant=constant
    )

    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)
