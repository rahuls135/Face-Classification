import math
import numpy as np


def bayes_classifier(testing_data, separated_data):

    def mean_var(data):
        means = np.mean(data, axis=0)
        variances = np.var(data, axis=0)
        return means, variances
    
    def gauss_pdf(x, mean, stdev):
        exp = math.exp((-1/2)*((x - mean)/(stdev + 0.01))**2)
        coefficient = (1/(stdev * math.sqrt(2*math.pi) + 0.01))
        return coefficient * exp
    
    means_and_vars = {}
    priors = {}
    for c, c_data in separated_data.items():
        # get mean and variances
        means_and_vars[c] = mean_var(c_data)
        # calculate priors
        priors[c] = len(c_data)/len(testing_data)
    
    # for c, c_data in separated_data.items():
    # get probabilities
    probs = {}
    for c, c_prior in priors.items():
        probs[c] = math.log(c_prior)
    
    # adding log-likelihoods
    for c in separated_data:
        for i in range(len(testing_data)):
            meanc = means_and_vars[c][0]
            varc = means_and_vars[c][1]
            pdf = gauss_pdf(testing_data[i], meanc, math.sqrt(varc))
            probs[c] += math.log(pdf + 0.00000001) 

    # prediction
    prediction = max(probs, key=probs.get)
    return prediction