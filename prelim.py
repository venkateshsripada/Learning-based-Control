import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import numpy
import csv
import random
import pdb
import numpy as np



# To filter the useless zero return values from the sonar
def isNotZero(x):
    return ( x != "0.0")


# Check if the row contains any nonzero return values besides the metadata
def isNotEmpty(row):
    for i in row[8:]:
        if i != "0.0":
            return True
    return False

def isNotShort(row):
    if sum(row[8:]) > 4:
        return True

def min_length(testdata):
    data = len(testdata)
    if data <= max(len(testdata)):
        return True
    else:
        for i in testdata:
            zero = np.zeros(len(row[testdata]), len(column[testdata]))
            testdata.append(zero)
    
# Read sonar .csv file, return semi-cleaned python list of floats
# TODO:  Use numpy arrays instead.  Bob, can you take a look at this?
def readSonarFile(filename):
    rawdata = []
    with open(filename, 'rt') as data_file:
        data_reader = csv.reader(data_file)
        next(data_reader, None)  # Ignore column headers
        for row in data_reader:
            # Keep only metadata and 'sum' return data.  The rest shouldn't exist
            # TODO:  Data length may change, check with bob about format shifts
            rawdata.append(row[0:320])

    # Ensure that we do not use all-zero rows
    # Training on all zeroes is basically worthless
    rawdata = filter(isNotEmpty, rawdata)

    # Ensure that all values are numeric, they're read as string by default
    # TODO:  Just let numpy handle this conversion gracefully, bah
    for i in range(0, len(rawdata)):
        rawdata[i] = map(float, rawdata[i])

    # Randomize order of data lines, helps prevent certain NN deadlocks
    # Note:  This is NOT shuffling the contents of each line (scan/metadata),
    # only the order in which these lines are presented
    random.shuffle(rawdata)

    # Ended up not useful, kept here for prosperity
    # Check if removing lines with not much data helps
    #rawdata = filter(isNotShort, rawdata)

    return rawdata


# Split data into metadata (sonar properties, object location)
# and sonar return data.  Warning:  Hardcoded BS
def splitDataTypes(rawdata):
    metadata = []
    testdata = []
    # TODO:  Ensure that there are always 9 metadata fields at the beginning.
    for row in rawdata:
        metadata.append(row[:9])
        testdata.append(row[9:])
    #put min_length
    return metadata, testdata

# Split a training set into training and validation sets
# where proportion is the fraction that is used for training (commonly 0.8)
def splitValidationSet(data, proportion=0.8):
    split_index = int(round(len(data) * proportion))
    train_data = data[:split_index]
    vali_data = data[split_index:]

    return train_data, vali_data


# Given metadata (or raw data), return an array where out[i] = raw[i] for categorization
# Used since sklearn expects a 1-D categories array
def makeCategoryArray(data):
    cat_array = []
    for entry in data:
        cat_array.append( entry[0] )

    return cat_array


# Generate some junk data for the 'not a shape' category
def makeJunkData(num_rows, num_bins = 312):
    junk_data = []
    for i in range(0, num_rows):
        randos = []
        for j in range(0, num_bins):  # TODO Hardcoded for testing
            randos.append(random.random())
        junk_data.append(randos)
    return junk_data


# Literally everything that was once in main.  Turn raw .csv into sklearn-ready numpy arrays
# 1- Read raw data, do some very basic preprocessing (str->float, drop useless lines)
# 2- Split raw data into metadata and actual sonar data
# 3- Create a 1-D category array out of the metadata
# 4- Split both the sonar data and its category array into training and validation sets
def processRawData(filename, train_prop = 0.8):
    # Read the file and get all of its data into a usable array
    raw = readSonarFile(filename)

    # Split metadata from sonar return data
    metadata, realdata = splitDataTypes(raw)

    # Make a category array from the metadata
    # realdata[i] has category metadata[i][0] = cat_data[i], this is only for numpy/sklearn's benefit
    category_data = makeCategoryArray(metadata)

    # Note:  Since raw was shuffled during the read process, this *should*
    # have a decent sample of each category.
    # If it doesn't, the fix is going to be a pain in the butt
    # TODO:  Verify that this assumption holds
    train_data, vali_data = splitValidationSet(realdata, train_prop)
    train_cats, vali_cats = splitValidationSet(category_data, train_prop)

    # Cast everything as numpy arrays (sklearn requires this)
    train_vals = np.array(train_data, dtype=np.float64)
    train_cats = np.array(train_cats, dtype=np.float64)

    vali_vals = np.array(vali_data, dtype=np.float64)
    vali_cats = np.array(vali_cats, dtype=np.float64)

    return train_vals, train_cats, vali_vals, vali_cats



# BEGIN MAIN

training_proportion = 0.8

processed_data = processRawData("test_both_targets.csv", training_proportion)

# Unnecessarily verbose, done for legibility
# Could just as easily call processed_data[N] in the mlp function calls
training_data_values       = processed_data[0]
training_data_categories   = processed_data[1]
validation_data_values     = processed_data[2]
validation_data_categories = processed_data[3]


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.001)

#pdb.set_trace()

# TODO:  Try adding noise, try remove zeros and pad as needed, try more data

mlp.fit(training_data_values, training_data_categories)
print("Training set score: %f" % mlp.score(training_data_values, training_data_categories))
print("Test set score: %f" % mlp.score(validation_data_values, validation_data_categories))



### OLD MAIN FILE ###

# # Read raw data
# rect_rawdata = readSonarFile("test.csv")
# rect_metadata, rect_testdata = splitDataTypes(rect_rawdata)
# circ_rawdata = readSonarFile("test_circle.csv")
# circ_metadata, circ_testdata = splitDataTypes(circ_rawdata)
# #both_rawdata = readSonarFile("test_both_targets.csv")
# #both_metadata, both_testdata = splitDataTypes(both_rawdata)
#
# #junk_rawdata = makeJunkData(100)
#
#
# # Split data into training and validation subsets
# rect_train, rect_vali = splitValidationSet(rect_testdata, 0.8)
# circ_train, circ_vali = splitValidationSet(circ_testdata, 0.8)
# #both_train, both_vali = splitValidationSet(both_testdata, 0.8)
# #junk_train, junk_vali = splitValidationSet(junk_rawdata, 0.8)
#
# # Make category arrays so the MLP is happy
# rect_train_cats = makeCategoryArray(rect_train, 0)
# rect_vali_cats  = makeCategoryArray(rect_vali, 0)
# circ_train_cats = makeCategoryArray(circ_train, 1)
# circ_vali_cats  = makeCategoryArray(circ_vali, 1)
# #both_train_cats = makeCategoryArray(both_train, 2)
# #both_vali_cats  = makeCategoryArray(both_vali, 2)
#
# #junk_train_cats = makeCategoryArray(junk_train, 2)
# #junk_vali_cats  = makeCategoryArray(junk_vali, 2)
#
# # Combine all of these into massive ugly arrays
# # train_data[i] should have category cats_data[i].  If it doesn't we screwed up
# train_vals = []
# train_cats = []
# train_vals.extend(rect_train);      train_vals.extend(circ_train)     ; #train_vals.extend(junk_train)
# train_cats.extend(rect_train_cats); train_cats.extend(circ_train_cats); #train_cats.extend(junk_train_cats)
# #train_vals.extend(both_train)
# #train_cats.extend(both_train_cats)
#
# vali_vals = []
# vali_cats = []
# vali_vals.extend(rect_vali)     ; vali_vals.extend(circ_vali)     ; #vali_vals.extend(junk_vali)
# vali_cats.extend(rect_vali_cats); vali_cats.extend(circ_vali_cats); #vali_cats.extend(junk_vali_cats)
# #vali_vals.extend(both_vali)
# #vali_cats.extend(both_vali_cats)