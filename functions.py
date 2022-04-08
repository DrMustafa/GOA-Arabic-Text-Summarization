import transform
import math
import json
import re
from collections import Counter

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    #print ("text=",text)
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)

def get_result(content_a, content_b):
    text1 = content_a
    text2 = content_b
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    cosine_result = get_cosine(vector1, vector2)
    return cosine_result

def ackley(solution, decode_func): 
    output=(f[0]+f[1]+f[2]+f[3]+f[4])*0.2   
    fitness=output
    finished = False
    return fitness, finished


def ackley_binary(binary):
    # Helpful functions from helpers are used to convert binary to floats
    x1 = transform.binary_to_float(binary[0:16], -5, 5)
    x2 =transform.binary_to_float(binary[16:32], -5, 5)
    return x1, x2

def ackley_real(values):
    return values

