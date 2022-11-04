function Classifier

    main()

end

function main()
    %{
    Goal : Run the test of the classifier
    Parameters : 
        None
    Return :
        None
    %}
    global nbWordVOCBuilding 
    global nbImagesVOCBuilding
    global nbImagesClsTraining 
    global nbImagesTesting
    global nbFdImages
    global fdAlgo
    global trainingSet
    global treshold


    % Parameters : 
    nbWordVOCBuilding = 200; % Vocabulary : K parameter in Kmeans in VOCBuilding
    nbImagesVOCBuilding = -1; % Number of images processed in VOCBuilding, Max value at -1
    nbImagesClsTraining = -1; % Number of images used to train the classifier, Max value at -1
    nbImagesTesting = -1; % Number of images used to test the whole algo, Max value at -1
    nbFdImages = 300; % Number of features descriptors per images
    fdAlgo = "SIFT"; % Method used to find the features : "SIFT", "ORB"
    trainingSet = "train"; % Which set is used
    treshold = 120000;
    
    % End of parameters

    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;

    % train and test classifier for each class
    for i=1:VOCopts.nclasses
        cls=VOCopts.classes{i};
        classifier=train(VOCopts,cls);                  % train classifier

        hour = fix(clock);
        fprintf("%dH:%dM:%dS - Testings\n", hour(4),hour(5),hour(6))
        test(VOCopts,cls,classifier);                   % test classifier
        hour = fix(clock);
        fprintf("%dH:%dM:%dS - End of Testings\n", hour(4),hour(5),hour(6))
        [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   % compute and display ROC
        if i<VOCopts.nclasses
            fprintf('press any key to continue with next class...\n');
            pause;
        end
    end
end

function classifier = train(VOCopts,cls)
    %{
    Goal : Implement the training of the classifier (according to the theory, it should be the VOC and the training part)
    Parameters : 
        VOCopts : The object from the VOC dev kit
        cls : String which contain the type of object we want to test
    Return : 
        classifier : This classifier object will be used to make the classifier works
    %}
    global fdAlgo

    % Load the name of the image for the selected set and the selected class
    [ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,"train"),'%s %d');
    % VOCabulary Building
    classifier.FD = VOCBuilding(VOCopts, ids, fdAlgo);
    hour = fix(clock);
    fprintf("%dH:%dM:%dS - Classifier training\n", hour(4),hour(5),hour(6))
    % Training of the model
    classifier.model = ClassifierTraining(VOCopts, classifier, ids);
end

function c = VOCBuilding(VOCopts, ids, fdAlgo)
    %{
    Goal : Implementation the Building of the vocabulary for buck of word
    Parameters : 
        VOCopts : The object from the VOC dev kit
        ids : The list of images used
    Return : 
        f : a nbWord of features (for Sift : nbWord by 128)
    %}
    global nbImagesVOCBuilding
    global nbWordVOCBuilding
    tic; % Fct to start the timer
    if nbImagesVOCBuilding == -1
        nbImagesVOCBuilding = length(ids);
    end
    drawnow;

    try 
        error("defecting to reading images")
        load(sprintf(VOCopts.exVOCpath,fdAlgo),'c');
        fprintf("Skipping the reading of the images \n")

    catch  
        hour = fix(clock);
        fprintf('total of images : %d Images used : %d\n', length(ids), nbImagesVOCBuilding);
        fprintf("%dH:%dM:%dS - Reading of the images\n", hour(4),hour(5),hour(6))
        for i=1:nbImagesVOCBuilding % For each name of image in the dataset
            % display progress
            if toc>1
                fprintf('train: %d/%d\n',i,nbImagesVOCBuilding);
                drawnow;
                tic;
            end
            try
                load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
            catch
                % Read the image, extract the features, and stores the features
                I=imread(sprintf(VOCopts.imgpath,ids{i})); % Lecture de l'image
                fd = extractfd(I, fdAlgo); % Extraction of features, return a tab of nbFeat * LenghtPerFeature (ex SIFT lenngth = 128)
                % fd = gpuArray(fd); % Using GPU | comment to use the CPU
                % features in a file previous line commented bc I try to save
                % the whole model in different states 
                save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
            end
            if i == 1 % In the case of the first iteration, where fd is undefined
                AllFeatures = fd; % Normal way
            else
                AllFeatures = [AllFeatures, fd]; 
            end
            
        end
         
        % Problem too much time, consider lowering the values
        % Maybe just ignore the warning, or sort the fd directly in extractfeature but can cause some problem
        % Maybe just too much value
        hour = fix(clock);
        fprintf("%dH:%dM:%dS - ", hour(4),hour(5),hour(6))
        fprintf("Clustering algo\n")
        % Hierarchical clustering
        % eucD = pdist(AllFeatures,'euclidean');
        % c = linkage(eucD,'average');
        % kmean clustering
        [c, ~] = vl_ikmeans(AllFeatures, nbWordVOCBuilding, 'Verbose');
        hour = fix(clock);
        fprintf("%dH:%dM:%dS - ", hour(4),hour(5),hour(6))
        fprintf("Algo finished\n")
        % We are only interested in C, which is a vector of K by 128
        save(sprintf(VOCopts.exVOCpath,num2str(fdAlgo)),'c'); % Save of the voc
    end
    

end

function model = ClassifierTraining(VOCopts, classifier, ids)
    %{
    Goal : Train the classifier
    Parameters : 
        VOCopts : The object from the VOC dev kit
        classifier : The classifier object which contains different item, obj.gt : if the image is from the class or not, obj.FD the Words features
        ids : List of image filename
    Return : 
        model : return the trained classifier
    %}
    global nbImagesClsTraining

    if nbImagesClsTraining == -1  
        nbImagesClsTraining = length(ids); % Number of image in the classifier training
    end
    nbWord = size(classifier.FD);
    nbWord = nbWord(2);
    X = zeros(nbImagesClsTraining, nbWord); % The histogram of all the trainning images
    y = zeros(nbImagesClsTraining, 1); % The Binary value of the image wether is from the class or not
    hour = fix(clock);
    fprintf("%dH:%dM:%dS - Reading of the images\n", hour(4),hour(5),hour(6))
    for i=1:nbImagesClsTraining
        histogram = SearchWord(VOCopts, ids{i},classifier.FD);
        X(i, 1:nbWord) = transpose(histogram);
        y(i,1) = classifier.gt(i); % Wether it is or not from the searched class
    end
    hour = fix(clock);
    fprintf("%dH:%dM:%dS - Training of the model\n", hour(4),hour(5),hour(6))

    model = fitcsvm(X,y); % Training of the SVM model
%     model =fitcsvm(X,y,'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'))

    hour = fix(clock);
    fprintf("%dH:%dM:%dS - End of training of the model\n", hour(4),hour(5),hour(6))


end

function test(VOCopts,cls,classifier)
    %{
    Goal : Test the classifier over a class of object 
    Parameters : 
        VOCopts : The object from the VOC dev kit
        cls : String which contain the type of object we want to test
        classifier : The classifier object which already contains a number of item during the training 
    Return : 
        Only display or write the result in a file
    %}
    global nbImagesTesting
    % load test set ('val' for development kit)
    [ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

    % create results file
    fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

    % classify each image
    tic;
    if nbImagesTesting == -1 
        nbImagesTesting = length(ids);
    end
    fprintf('%s : total of images : %d Images used : %d\n', cls, length(ids), nbImagesTesting);
    drawnow;
    for i=1:nbImagesTesting
        % display progress
        if toc>1
            fprintf('%s: test: %d/%d\n',cls,i,nbImagesTesting);
            drawnow;
            tic;
        end
        
        % Read the image and compute the histogram
        histogram = SearchWord(VOCopts, ids{i}, classifier.FD);

        % compute confidence of positive classification
        c=classify(VOCopts,classifier,histogram);
        
        % write to results file
        fprintf(fid,'%s %f\n',ids{i},c);
    end

    % close results file
    fclose(fid);

end

function total = matchFeaturesLocal(ft1, ft2, fdAlgolc)
%     ft2 tab de features
    global fdAlgo
    if isnumeric(fdAlgolc)
        fdAlgolc = fdAlgo;
    end
    global treshold
    min=9999999;
    total = 0;
    sz = size(ft2);
    for i=1:sz(2)
        if fdAlgolc == "RGB" 
            if ft1 == ft2(i)
                total = total + 1;
            end
        else
            % ip = matchFeatures(transpose(ft1), transpose(ft2(i,:))); %
            % Error it will furnish a 0 by 2 matrix even if the result is
            % the same
            [dist,sc]= vl_ubcmatch(uint8(ft1),ft2(:,i)); % sc is theeuclidiandistancebtwpoints
            if sc < treshold
                total = total + 1;
            end
            if  min>sc
                min=sc;
            end
        end
    end
end

function histogram = SearchWord(VOCopts, ids, Bow)
    %{
    Goal : for a given image, and a given buck of word, this function will search those words in the image and return a histogram
    Should work, tested on basic situation
    Parameters : 
        ids : Name of theimage
        Bow : Buck of word an array of k by 128 for SIFT
    Return : 
        histogram : The histogram of the BOW (normalized)
    %}
    % Parameters : 
    try
        load(sprintf(VOCopts.exfdpath,ids),'fd');
    catch
        I=imread(sprintf(VOCopts.imgpath,ids));
        fd = extractfd(I, -1);
        save(sprintf(VOCopts.exfdpath,ids),'fd');
    end
    szBw = size(Bow);
    histogram = zeros(szBw(2),1);
    cum = 0;
    
    for i=1:szBw(2)
        histogram(i) = matchFeaturesLocal(Bow(:,i), fd, -1);
        cum = cum + histogram(i);
    end


    
    % Pair of 2 column to match between features
    
    % Adding 1 for each occurence in the indexPairs variable

    % Normalization
    tmp = zeros(szBw(2),1);
    for i=1:length(histogram)
        if histogram(i)~= 0
            tmp(i) = histogram(i)/cum;
        else
            tmp(i) = 0;
        end

    end
    histogram = tmp;
    

end

function I = preProcessingImages(I)
    %{
    Goal : Preprocessing of the image, to add different filter to the image before being processed
    Parameters : 
        I : The image from which we will extract our features
    Return : 
        I : The modified images, it have to be in 2D, monochannel with the function RGB2GRAY
    %}
    
    I = rgb2gray(I);
    % I = edge(I,"log", [], 0.2);

end

function fd = extractfd(I, fdAlgolc)
    %{
    Goal : Extract all the features descriptors of the Image I
    Parameters : 
        I : The image from which we will extract our features
    Return : 
        fd : The features descriptors of the image I
    %}
    global nbFdImages;
    global fdAlgo
    if isnumeric(fdAlgolc)
        fdAlgolc = fdAlgo;
    end
         
    fd = [];                       
    I = preProcessingImages(I);
    if fdAlgolc == "SIFT"
        I = single(I);
        [f,fd] = vl_sift(I);
        % fd = transpose(fd);
        % fd = double(fd);
    elseif fdAlgolc == "ORB"
        points = detectORBFeatures(I);
        fd = points.selectStrongest(nbFdImages);
        fd = extractFeatures(I,fd).Features;
        fd = double(fd);
    elseif fdAlgolc == "RGB"
        fd = [[mean(I, "all")]];
    else 
        error("Algo not implemented : "+fdAlgolc)
    end
end    

function c = classify(VOCopts,classifier,histogram)

    %{
    Goal : Use the classifier trained
    Parameters : 
        VOCopts : The object from the VOC dev kit
        classifier : The classifier object which already contains a number of item during the training 
        histogram : All the features descriptors
    Return : 
        c : The confidence for the object
    %}
    % Here implement the classifier (SVM, Adaboost, etc...)

    [label,score] = predict(classifier.model, transpose(histogram));
    c = score(2);

end