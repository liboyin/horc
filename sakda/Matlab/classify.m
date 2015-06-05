function [ Index ] = classify( filename, descriptor_bank )
%% Testing Images Extract Sift
    TOTAL_CAT=size(descriptor_bank,2);
    IMG_IN_CAT=length(fieldnames(descriptor_bank))-1;
    kernel = -1 * ones(3)/9;
    list=cell(IMG_IN_CAT,1);
    for i=1:IMG_IN_CAT
        list{i}=genvarname(sprintf('desc%d',i));
    end
    I = imreadbw(filename);
    I = imfilter(I,kernel);
    [frames,descriptors] = sift(I,'edgeThreshold',12);
    
    match=cell(IMG_IN_CAT,1);
    matches=zeros(1,TOTAL_CAT,1);
    for testwithcat=1:TOTAL_CAT
        for eachfile=1:IMG_IN_CAT
            match{eachfile} = siftmatch(descriptors, descriptor_bank(testwithcat).(list{eachfile}));
            matches(testwithcat) = matches(testwithcat) + size(match{eachfile},2);
        end
    end

%% Analyse Result
fprintf('############## RESULTS ###############\n');
[Max, Index]=max(matches,[],2);

