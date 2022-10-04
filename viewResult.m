function ReadResult(file)
    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);
    
    % initialize VOC options
    VOCinit;
    for i=1:length(VOCopts.classes)
        cls=VOCopts.classes{i};
        [fp,tp,auc]=VOCroc(VOCopts,file,cls,true);   % compute and display ROC
        if i<VOCopts.nclasses
            fprintf('press any key to continue with next class...\n');
            pause;
        end
    end

end