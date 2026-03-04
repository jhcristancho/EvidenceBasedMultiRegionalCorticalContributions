clear
close all
clc
beep off

directory = [pwd filesep 'data'];
% snames = {'bp' 'cc' 'ht' 'jc' 'jp' 'mv' 'wc' 'wm' 'zt'};
snames = {'cc'};

Acc = zeros(5,length(snames));
for subject = 1:length(snames)
   
    % keep basic variables
    clearvars -except subject snames directory Acc
    fprintf('Subject %d \n',subject)
    
    % Load data
    fprintf('Analysing subject %d \n',subject)
    fprintf('loading data ... \n')
    
    %load dataet
    load([directory '\' snames{subject} '\' snames{subject} '_fingerflex']);
    load([directory '\' snames{subject} '\' snames{subject} '_stim'])    
    fs = 1000;
    
    
    %detrend data and do CAR
    data = detrend(double(data));
    data = bsxfun(@minus,data,mean(data,2));
    lf = downsample(prep_bpfilter(data,4,.05,8,fs),50); %Lowpass data
    hg = downsample(prep_lpfilter(abs(hilbert(prep_bpfilter(data,8,70,170,fs))),4,8,fs),50); %HG data
    labels = downsample(stim,50); %Labels
    fs = fs/10;
    
    %%
    ind = labels>=0; % remove values -2 and -1 from the labels which are unknown
    feats = zscore(hg(ind,:),0,1);
    targets = labels(ind,:);
    
    nfolds = 5;
    segInd = floor(linspace(1,size(feats,1),nfolds+1)); %Separator of segments
    dataSeg = cell(length(segInd)-1,1); %Data
    targetsSeg = cell(length(segInd)-1,1); %Labels
    for i = 1:length(segInd)-1
        dataSeg{i,1} = feats(segInd(i):segInd(i+1),:)';
        targetsSeg{i,1} = int32(targets(segInd(i):segInd(i+1),:))';
    end
    
    cvInd = 1:nfolds;
    
    % Model parameters
    toolBoxParams = struct();    
%     toolBoxParams.modelType = 'ldcrf';
    toolBoxParams.modelType = 'crf';
    toolBoxParams.optimizerType = 'bfgs';
%     toolBoxParams.nbHiddenStates = 2;
    toolBoxParams.nbHiddenStates = 0;
%     toolBoxParams.windowSize = 4;
    toolBoxParams.windowSize = 0;
    
    params = struct();    
    params.debugLevel = 0;
    params.regularizationL2 = 0;
    params.regularizationL1 = 0;
    params.weightsInitType = 'RANDOM';  % ZERO, CONSTANT, RANDOM, GAUSSIAN
    params.maxIterations = 50;
    params.minRangeWeights = -2;
    params.maxRangeWeights = 2;
    %params.initWeights = init;
    %%
    clc
    for m = 1:max(cvInd)
        trainData = dataSeg(cvInd~=m);
        testData = dataSeg(cvInd==m);
        
        trainTargets = targetsSeg(cvInd~=m);
        testTargets = targetsSeg(cvInd==m);
        
        % Train Model
        matHCRF('createToolbox',toolBoxParams.modelType,toolBoxParams.optimizerType,toolBoxParams.nbHiddenStates,toolBoxParams.windowSize);
%         matHCRF('createToolbox',toolBoxParams.modelType,toolBoxParams.optimizerType,toolBoxParams.nbHiddenStates,toolBoxParams.windowSize);
        matHCRF('setData',trainData,trainTargets);
        paramsNames = fields(params);
        for j = 1:size(paramsNames,1)
            matHCRF('set',paramsNames{j,1},params.(paramsNames{j,1}));
        end
        matHCRF('train');
        
        %[model, featureDefinition] = matHCRF('getModel');
        matHCRF('setData',testData,testTargets);
        matHCRF('test');
        ll=matHCRF('getResults');
        
        [val, ind] = max(ll{1},[],1);
        Acc(m,subject) = mean((ind-1)'==testTargets{1}');
        
    end
    save([directory filesep snames{subject} filesep snames{subject} '_res_CRF_all_profe_windowSize_0.mat'],'Acc','dataSeg','targetsSeg');
 end