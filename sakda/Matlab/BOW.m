clear all

% Define start category, 1=start from image001, 2=start from image006, and so on 
START_CAT=1;

% Define end category, 2=end at image004, 3=end at image009, and so on
END_CAT=50;

% Define a number of cluster
nc=floor((END_CAT-START_CAT)*1.6);

% Define a number of descriptors picked from each training image
nd=5;

% Define where the images are located
imgFolder = fullfile('./dataset/');
imgSets = dir(imgFolder);

% remove file name: . and .. from image set
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
        if eachfile == tester
            testingSet=[testingSet dir(fullfile('./dataset/',imgSets((4*(cat-1))+eachfile).name))];
        else
            trainingSet=[trainingSet dir(fullfile('./dataset/',imgSets((4*(cat-1))+eachfile).name))];
        end
    end
end

%% Training Images Extract Sift and Store in Structure (category,desc1,desc2,desc3)
fprintf('############## LEARNING ##############\n');
descriptor_bank=struct;
frame_bank=struct;
parfor cat=1:END_CAT-START_CAT
    descriptors=cell(1,3);
    frames=cell(1,3);
    for eachfile=1:3
        filename=fullfile('./dataset/',trainingSet((cat-1)*3+eachfile).name);
        fprintf('Sifting File: %s\n', filename);
        I = imreadbw(filename);
        [frames{eachfile},descriptors{eachfile}] = sift(I);
    end
    descriptor_bank(cat).category=cat+START_CAT;
    descriptor_bank(cat).desc1=descriptors{1}';
    descriptor_bank(cat).desc2=descriptors{2}';
    descriptor_bank(cat).desc3=descriptors{3}';
    frame_bank(cat).category=cat+START_CAT;
    frame_bank(cat).frm1=frames{1};
    frame_bank(cat).frm2=frames{2};
    frame_bank(cat).frm3=frames{3};
end

%% Sort scale of frames from large to small
d=zeros(3*nd*(END_CAT-START_CAT),128);
for cat=1:END_CAT-START_CAT
    [Y1,I1]=sort(frame_bank(cat).frm1(3,:),2,'descend');
    [Y2,I2]=sort(frame_bank(cat).frm2(3,:),2,'descend');
    [Y3,I3]=sort(frame_bank(cat).frm3(3,:),2,'descend');
    
    % Pick descriptors from top nd frames scale
    for eachd=1:nd
        d(eachd+(cat-1)*3*nd,:)=descriptor_bank(cat).desc1(I1(eachd+3),:);
        d(eachd+nd+(cat-1)*3*nd,:)=descriptor_bank(cat).desc2(I2(eachd+3),:);
        d(eachd+(nd*2)+(cat-1)*3*nd,:)=descriptor_bank(cat).desc3(I3(eachd+3),:);
    end
end

%% Find k-means and generate bag of words
trainingH=zeros(nc,3*(END_CAT-START_CAT));
[clusters, centers]=kmeans(d,nc);
list=genvarname({'desc','desc','desc','desc'});
for cat=1:END_CAT-START_CAT
    for eachfile=1:3
        distances=dist2(descriptor_bank(cat).(list{eachfile+1}),centers);
        [minvals, mininds] = min(distances, [], 2);
        trainingH(:,(cat-1)*3+eachfile)=histc(mininds,1:nc);
    end
end
fprintf('################ END #################\n\n');

%% Testing Images Extract Sift
fprintf('############ CLASSIFYING #############\n');
testingH=zeros(nc,END_CAT-START_CAT);
parfor cat=1:END_CAT-START_CAT
    filename=fullfile('./dataset/',testingSet(cat).name);
    fprintf('Testing File: %s\n', filename);
    I = imreadbw(filename);
    [frames,descriptors] = sift(I);
    distances=dist2(descriptors',centers);
    [minvals, mininds] = min(distances, [], 2);
    testingH(:,cat)=histc(mininds,1:nc);
end
distances=zeros(END_CAT-START_CAT,3*(END_CAT-START_CAT));
for cat=1:END_CAT-START_CAT
    for eachbag=1:3*(END_CAT-START_CAT)
        distances(cat,eachbag)=dist2(trainingH(:,eachbag)',testingH(:,cat)');
    end
end

%% Find closest word
[Min, Index]=min(distances,[],2);
result=ceil(Index/3);
error=0;
for cat=1:END_CAT-START_CAT
    fprintf('File: %s\n',testingSet(cat).name);
    fprintf('Predict Category: %d, Filename Range: image%03d - %03d\n\n',result(cat),((result(cat)-1)+START_CAT-1)*5+1,((result(cat)-1)+START_CAT-1)*5+4);
    if cat~=result(cat)
        error=error+1;
    end
end
fprintf('Error: %d, Percentage: %2.2f%%\n',error,error/END_CAT*100);
fprintf('################ END #################\n');

figure(1);
bar(1:nc,trainingH,'histc')
figure(2);
bar(1:nc,testingH,'histc')
