import math

predictions = {"joie": 0.9, "anxiété": 0.1, "tristesse": 0.0, "colère": 0.0, "fatigue": 0.1, "peur": 0.1}

def softmax(predictions):
    output = {}
    for sentiment, predicted_value in predictions.items():
        output[sentiment] = math.exp(predicted_value*10) / sum(math.exp(value*10) for value in predictions.values())
    return output

print(softmax(predictions))