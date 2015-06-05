clear all

% Set the start category
START_CAT=1;

% Set the end category
END_CAT=50;

% Set kernel
kernel = fspecial('gaussian',3);

% Set image path
imgFolder = fullfile('./dataset/','*.JPG');
imgSets = dir(imgFolder);

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

%% Training Images Extract Sift and Store in Structure (category,desc1,desc2,desc3)
fprintf('############## LEARNING ##############\n');
descriptor_bank=struct;
parfor cat=1:(END_CAT-START_CAT)+1
    descriptors=cell(1,3);
    for eachfile=1:3
        filename=fullfile('./dataset/',trainingSet((cat-1)*3+eachfile).name);
        fprintf('Sifting File: %s\n', filename);
        I = imreadbw(filename);
        I = imfilter(I,kernel);
        I = imcomplement(I);
        [frames,descriptors{eachfile}] = sift(I,'edgeThreshold',12);
    end
    descriptor_bank(cat).category=cat+START_CAT;
    descriptor_bank(cat).desc1=descriptors{1};
    descriptor_bank(cat).desc2=descriptors{2};
    descriptor_bank(cat).desc3=descriptors{3};
end
fprintf('################ END #################\n\n');

%% Testing Images Extract Sift
fprintf('############ CLASSIFYING #############\n');
matches=zeros(END_CAT-START_CAT+1);
parfor cat=1:(END_CAT-START_CAT)+1
    filename=fullfile('./dataset/',testingSet(cat).name);
    fprintf('Testing File: %s\n', filename);
    I = imreadbw(filename);
    I = imfilter(I,kernel);
    I = imcomplement(I);
    [frames,descriptors] = sift(I,'edgeThreshold',12);
    for testwithcat=1:(END_CAT-START_CAT)+1
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
for cat=1:(END_CAT-START_CAT)+1
    fprintf('File: %s\n',testingSet(cat).name);
    fprintf('Predict Category: %d, Filename Range: image%03d - %03d\n\n',Index(cat),((Index(cat)-1)+START_CAT-1)*5+1,((Index(cat)-1)+START_CAT-1)*5+4);
    if cat~=Index(cat)
        error=error+1;
    end
end
fprintf('Error: %d, Percentage: %2.2f%% correct!!!\n',error,100-(error/(END_CAT-START_CAT+1)*100));
fprintf('################ END #################\n');