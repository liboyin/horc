function [ descriptor_bank ] = learn( imgFolder, IMG_IN_CAT, TOTAL_CAT )
    kernel = -1 * ones(3)/9;
    imgSets = dir(imgFolder);
    if strcmp(imgSets(1).name,'.')
        imgSets(1) = [];
    end
    if strcmp(imgSets(1).name,'..')
        imgSets(1) = [];
    end
    fprintf('############## LEARNING ##############\n');
    descriptor_bank=struct;
    list=cell(IMG_IN_CAT,1);
    for i=1:IMG_IN_CAT
        list{i}=genvarname(sprintf('desc%d',i));
    end
    parfor cat=1:TOTAL_CAT
        descriptors=cell(1,IMG_IN_CAT);
        for eachfile=1:IMG_IN_CAT
            filename=fullfile(imgFolder,imgSets((cat-1)*IMG_IN_CAT+eachfile).name);
            fprintf('Sifting File: %s\n', filename);
            I = imreadbw(filename);
            I = imfilter(I,kernel);
            [frames,descriptors{eachfile}] = sift(I,'edgeThreshold',12);

        end
        descriptor_bank(cat).category=cat;
        for eachfile=1:IMG_IN_CAT
            descriptor_bank(cat).(list{eachfile})=descriptors{eachfile};
        end
    end
    fprintf('################ END #################\n\n');
end

