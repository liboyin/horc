File BOW.m:

- This file is Bag-of-Word matlab script. 
- It read images from ./dataset folder
- This file use Andrea Vedaldi SIFT implementation.
- In order to run this script, please download MATLAB/C implementation of the SIFT detector and descriptor from http://www.robots.ox.ac.uk/~vedaldi/code/sift.html and add path to it.
- This script uses "parfor" which runs multi-threads.

File SiftMatch.m is SIFT match classifitation script.

- It read images from ./dataset folder
- It take 3 images for trainers and 1 for tester in each class randomly.
- This file use Andrea Vedaldi SIFT implementation.
- In order to run this script, please download MATLAB/C implementation of the SIFT detector and descriptor from http://www.robots.ox.ac.uk/~vedaldi/code/sift.html and add path to it.
- This script uses "parfor" which runs multi-threads.

File learn.m is a learn function
- Generate descriptor bank used for classifying
- Usage: bank = learn(image_folder, number_of_images_per_category, total_number_of_category)
- Example: bank = learn('dataset', 4, 50)

File classify.m is a classify function
- Predict a image's class
- Usage: answer = classify(filename, descriptor_bank)

File dist2.m is taken from "Roland Bunschoten"