import matplotlib.pyplot as plt


def plotTrainTestPerformance(train, test, change, x_values=[]):
    train[:] = [1 - x for x in train]
    test[:] = [1 - x  for x in test]
    if not x_values:
        plt.plot(train)
        plt.plot(test)
    else:
        plt.plot(x_values, train)
        plt.plot(x_values, test)
    plt.plot(train)
    plt.plot(test)
    axes = plt.gca()
    #axes.set_ylim([0,0.05 ])
    plt.xlabel(change)
    plt.ylabel('Error')
    plt.legend(['Training', 'Testing'], loc=1)
    plt.show()

    