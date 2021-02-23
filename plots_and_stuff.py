import matplotlib.pyplot as plt


def plotTrainTestPerformance(train, test, change):
    train[:] = [1 - x for x in train]
    test[:] = [1 - x  for x in test]
    plt.plot(train)
    plt.plot(test)
    axes = plt.gca()
    #axes.set_ylim([0,0.05 ])
    plt.xlabel(change)
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Testing'], loc=1)
    plt.show()