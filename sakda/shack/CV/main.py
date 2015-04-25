from os.path import exists
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from cPickle import dump, load, HIGHEST_PROTOCOL
import random

import sift
import libsvm

PRE_ALLOCATION_BUFFER = 1000
K_THRESH = 1

outputdir = '../CVdatafiles'
codebook_file = outputdir + '/codebook.file'
training_histogram_file = outputdir + '/trainingdata.svm'
testing_histogram_file = outputdir + '/testingdata.svm'
predict_file = outputdir + '/trainingdata.svm.prediction'

imagedir = '../dataset'
basename = imagedir + '/image'
ext = '.JPG'
maxfile = 100 #last file is image249.JPG, enter 250 for train-test all of them. To train-test some of them, change to a smaller number (dividable by 5)

def get_categories():
    cats = [i for i in range(1,1+maxfile/5)]
    return cats

def get_train_imgfiles(cat):
    catfiles = [basename+str((cat-1)*5+i).zfill(3)+ext for i in range(1,5)]
    i = random.randint(0, 3)
    del catfiles[i] #remove one image for being the tester
    return catfiles

def get_test_imgfiles(cat,trainfiles,testfiles):
    allcatfiles = [basename+str((cat-1)*5+i).zfill(3)+ext for i in range(1,5)]
    #Find an image that is not in the training files
    for file in trainfiles:
        allcatfiles.remove(file)
    testfiles.append(allcatfiles[0])
    return testfiles

def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'
        if exists(features_fname) == False:
            print "calculating sift features for", fname
            sift.process_image(fname, features_fname)
        print "gathering sift features for", fname
        locs, descriptors = sift.read_features_from_file(features_fname)
        print descriptors.shape
        all_features_dict[fname] = descriptors
    return all_features_dict

def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(1, nwords+1):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)

if __name__ == '__main__':
    print "################# Learning Process Start ################"
    print "## loading the images and extracting the sift features"
    cats = get_categories()
    ncats = len(cats)
    print "searching for folders at " + imagedir
    if ncats < 1:
        raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
    print "found following folders / categories:"
    print cats
    print "---------------------"
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    test_files = []
    for cat in cats:
        cat_files = get_train_imgfiles(cat)
        test_files = get_test_imgfiles(cat,cat_files,test_files)
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = cat
        for i in cat_files:
            all_files_labels[i] = cat

    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    with open(codebook_file, 'wb') as f:
        dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    print "---------------------"
    print "## compute the visual words histograms for each image"
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          all_files,
                          all_word_histgrams,
                          training_histogram_file)

    print "---------------------"
    print "## train svm"
    c, g, rate, model_file = libsvm.grid(training_histogram_file,
                                         png_filename='grid_res_img_file.png')

    print "--------------------"
    print "## outputting results"
    print "model file: " + outputdir + model_file
    print "codebook file: " + codebook_file
    print "category      ==>  label"
    for cat in cat_label:
        print '{0:13} ==> {1:6d}'.format(cat, cat_label[cat])

    print "################# Learning Process End ##################"
    print "################# Testing Process Start #################"
    all_files_labels = {}
    all_features = extractSift(test_files)
    for i in test_files:
        all_files_labels[i] = 0  # Put as unknown first

    print "---------------------"
    print "## loading codebook from " + codebook_file
    with open(codebook_file, 'rb') as f:
        codebook = load(f)

    print "---------------------"
    print "## computing visual word histograms"
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    nclusters = codebook.shape[0]
    writeHistogramsToFile(nclusters,
                      all_files_labels,
                      test_files,
                      all_word_histgrams,
                      testing_histogram_file)

    print "---------------------"
    print "## test data with svm"
    libsvm.test(testing_histogram_file, model_file)
    print "################# Testing Process End ###################"
    print "######################## Results ########################"
    with open(predict_file) as predict:
        content = predict.readlines()
    total_line = len(content)
    for line in range(0,total_line):
        cat = int(content[line])
        print "Category:", cat
        # Train file shows a range of filenames in that category, but one of them is not used in training process.
        # Not used one is the tester. The filename must be in the range of filenames in that category, means correct category
        print "Train Files: image", str(5*(cat-1)).zfill(3), "-", str(5*(cat-1)+4).zfill(3)
        print " Test Files:", test_files[line]
        print ""
    print "########################## End ##########################"
