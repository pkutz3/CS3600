from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData,buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

def results(results):
    mx = max(results)
    avg = average(results)
    sd = stDeviation(results)
    print("Max:     {}".format(mx))
    print("Average: {}".format(avg))
    print("SD:      {}".format(sd))
    return [mx, avg, sd]

def q5():
    print("Q5")
    print("Pen Data:")
    print("---------------------------")
    pResults = []
    for i in range(5):
        print("Run {}".format(i))
        pResults.append(testPenData()[1])
    print("Pen Data:")
    print("---------------------------")
    results(pResults)
    print("---------------------------")

    print("Car Data:")
    print("---------------------------")
    cResults =  []
    for i in range(5):
        print("Run {}".format(i))
        cResults.append(testCarData()[1])
    print("Car Data:")
    print("---------------------------")
    results(cResults)
    print("---------------------------")

def q6():
    print("Q6")
    print("Pen Data:")
    print("---------------------------")
    fullResults = []
    for i in range(0, 41, 5):
        pResults = []
        for j in range(5):
            pResults.append(testPenData([i])[1])
        print("Pen Data Perceptron Count {}".format(i))
        res = results(pResults)
        res.insert(0, i)
        fullResults.append(res)
    print("---------------------------")
    print("Pen Full Data:")
    print(fullResults)
    print("---------------------------")

    print("Car Data:")
    print("---------------------------")
    for i in range(0, 41, 5):
        cResults = []
        for j in range(5):
            cResults.append(testCarData([i])[1])
        print("Car Data Perceptron Count {}".format(i))
        res = results(cResults)
        res.insert(0, i)
        fullResults.append(res)
    print("---------------------------")
    print("Car Full Data:")
    print(fullResults)
    print("---------------------------")

xorData = ([([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0])], [([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0])])
def testXorData(hiddenLayers = [16]):
    return buildNeuralNet(xorData, alpha=.6, 
    weightChangeThreshold=.0000001, maxItr = 10000,hiddenLayerList = hiddenLayers)

def q7():
    print("Q7")
    print("XOR Data:")
    print("---------------------------")
    totalResults = {}
    for i in range(5):
        totalResults["Perceptron Count: " + str(i)] = 0

    for _ in range(30):
        for j in range(0, 5, 1):
            xResults = []
            for i in range(5):
                print("Run {}".format(i))
                xResults.append(testXorData([j])[1])
            totalResults["Perceptron Count: " + str(j)] += average(xResults)
    print("XOR Data:")
    print("---------------------------")
    for i in range(5):
        totalResults["Perceptron Count: " + str(i)] /= float(30)
    print(totalResults)
    print("---------------------------")

extraData = buildExamplesFromExtraData()
def testExtraData(hiddenLayers=[16]):
    return buildNeuralNet(extraData, maxItr = 5000,hiddenLayerList = hiddenLayers)

def q8(sessions=10, hiddenLayers=20):
    print("Q8")
    print("Extra Data:")
    print("---------------------------")
    exResults = []
    totalResults = 0
    for _ in range(sessions):
        for i in range(5):
            print("Run {}".format(i))
            exResults.append(testExtraData([hiddenLayers])[1])
        totalResults += average(exResults)
    print("Extra Data:")
    print("---------------------------")
    results(exResults)
    print("Overall average: {}".format(totalResults / float(sessions)))
    print("---------------------------")

#q5()
#q6()
#q7()
#q8(sessions=10,hiddenLayers=20)
