clear all

START_CAT=1;
END_CAT=50;

kernel = -1 * ones(3)/9;
imgFolder = fullfile('./dataset/');
imgSets = dir(imgFolder);
if (imgSets(1).name == '.')
   imgSets(1) = [];
end
if strcmp(imgSets(1).name,'..')
   imgSets(1) = [];
end

%% Separate images
trainingSet=[];
testingSet=[];
for cat=START_CAT:END_CAT
    tester = randi([1,4]);
    for eachfile=1:4
        i=1;
        if eachfile == tester
            testingSet=[testingSet dir(fullfile('./dataset/',imgSets((4*(cat-1))+eachfile).name))];
        else
            trainingSet=[trainingSet dir(fullfile('./dataset/',imgSets((4*(cat-1))+eachfile).name))];
            i=i+1;
        end
    end
end
testingSet=[testingSet dir(fullfile('./dataset/',imgSets(201).name))];

%% Training Images Extract Sift and Store in Structure (category,desc1,desc2,desc3)
fprintf('############## LEARNING ##############\n');
descriptor_bank=struct;
parfor cat=1:END_CAT-START_CAT
    descriptors=cell(1,3);
    for eachfile=1:3
        filename=fullfile('./dataset/',trainingSet((cat-1)*3+eachfile).name);
        fprintf('Sifting File: %s\n', filename);
        I = imreadbw(filename);
%        I = imreadbw(filename);
        I = imfilter(I,kernel);
        %I = imcrop(I, [80,30,180,180]);
        %I = imcrop(I, [50,20,220,200]);
        [frames,descriptors{eachfile}] = sift(I,'EdgeThreshold',12);
    end
    descriptor_bank(cat).category=cat+START_CAT;
    descriptor_bank(cat).desc1=descriptors{1};
    descriptor_bank(cat).desc2=descriptors{2};
    descriptor_bank(cat).desc3=descriptors{3};
end
fprintf('################ END #################\n\n');

%% Testing Images Extract Sift
fprintf('############ CLASSIFYING #############\n');
matches=zeros((END_CAT-START_CAT)+2);
parfor cat=1:(END_CAT-START_CAT)+2
    filename=fullfile('./dataset/',testingSet(cat).name);
    fprintf('Testing File: %s\n', filename);
     I = imreadbw(filename);
%    I = rgb2gray(imread(filename));
    I = imfilter(I,kernel);
    %I = imcrop(I, [80,30,180,180]);
    %I = imcrop(I, [50,20,220,200]);
    [frames,descriptors] = sift(I,'EdgeThreshold',12);
    for testwithcat=1:END_CAT-START_CAT
        matches1 = siftmatch(descriptors, descriptor_bank(testwithcat).desc1);
        matches2 = siftmatch(descriptors, descriptor_bank(testwithcat).desc2);
        matches3 = siftmatch(descriptors, descriptor_bank(testwithcat).desc3);
        %fprintf('cat: %3d, testwith: %3d, matches: %3d, %3d, %3d\n',cat,testwithcat,size(matches1,2),size(matches2,2),size(matches3,2));
        matches(cat,testwithcat)=size(matches1,2)+size(matches2,2)+size(matches3,2);
        %matches(cat,testwithcat)=max([size(matches1,2), size(matches2,2), size(matches3,2)]);
    end
end
fprintf('################ END #################\n\n');








%% Analyse Result
fprintf('############## RESULTS ###############\n');
[Max, Index]=max(matches,[],2);
error=0;
for cat=1:(END_CAT-START_CAT)+2
    fprintf('File: %s\n',testingSet(cat).name);
    fprintf('Predict Category: %d, Filename Range: image%03d - %03d\n\n',Index(cat),((Index(cat)-1)+START_CAT-1)*5+1,((Index(cat)-1)+START_CAT-1)*5+4);
    if cat~=Index(cat)
        error=error+1;
    end
end
fprintf('Error: %d, Percentage: %2.2f%% correct!!!\n',error,100-(error/(END_CAT-START_CAT)*100));
fprintf('################ END #################\n');