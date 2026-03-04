% function call_subs_processing
% executes the entire signal processing, etc for each motor subject
clear all
close all
clc

% if matlabpool('size') == 0
%     matlabpool('open');
% end


%% add path
% addpath dc_files
warning('off','signal:psd:PSDisObsolete'); %annoying
warning('off','MATLAB:pack:InvalidInvocationLocation'); %annoying

%% define subject and rhythm frequency range ("bands")
subjects=[
    'cc';...
    'bp';...
    'zt';...
    'jp';...
    'ht';...
    'mv';...
    'wc';...
    'wm';...
    'jc';...
    ];

%%
Fs = 1000; %Frequency sample
maxStates = 0;
Type = 'HG';
% Type = 'LP';
% Type = 'HG_LP';

% lengthTrial = round(0.9*Fs); %Original length of trial: 0.9 seconds
% percentBaseline = 1.5*0.2*Fs / lengthTrial; %In percentage. 
% It is the percentage of the total trial before the onset. The actual
% value of the baseline is 300 ms in the original length of trial.
% segment2chop = 0.5*0.2*Fs; %It is the lenght of the chopper for each extreme of a trial (see below)

for q=1:size(subjects,1)
    subject=subjects(q,:);
    disp(['Getting data for subject ' num2str(q) ' (' subject ') of ' num2str(size(subjects,1)) '...'])
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %     Checklist for each dataset:
    % % Steps 1-3 done ahead of time, all variables in file 'data/#subject#_fingerflex.mat
    % % 1 - reject bad channels - "data" variable has this already removed
    % % 2 - get rendering and locations - brain rendering structure is in "brain" variable
    % % 3 - assign electrodes to anatomic clusters - positions are in "locs" variable (channel number x 3), with labels in "elec_regions"
    % %         translation code for "elec_labels":
    %     1 ? dorsal M1
    %     2 - rolandic
    %     3 ? dorsal S1
    %     4 ? ventral sensorimotor (M1+ S1)
    %     6 ? frontal (non-rolandic)
    %     7 ? parietal (non-rolandic)
    %     8 ? temporal
    %     9 ? occipital    
    
    load(['data' filesep subject filesep subject '_stim'],'stim'); %Everything starts here :)
    
    load(['data' filesep subject filesep subject '_fingerflex']); 
    
    % [numData,numCol] = size(data);    
%     nonSM = or(elec_regions > 4,elec_regions < 1);
%     data(:,nonSM) =  []; %Only takes electrode in SM 
    data = detrend(data); %Detrending
    data = ref_car(data); %Common Average Removal 
%     data = data';
    
    deg_filter = 4;
%     deg_filter = 10;
    lp_freq = 20;
    disp('Filtering...');
    dataTrial_HG = [];
    sttimi = [];
    if strfind(Type,'LP') > 0
        tmpMat = prep_bpfilter(data,deg_filter,0.05,lp_freq,Fs); %Lowpass band
        dataTrial_HG = cat(1,dataTrial_HG,zscore(tmpMat,0,1));
        sttimi = cat(1,sttimi,stim);
    end
    
    if strfind(Type,'HG') > 0
        dataTrial_filtered = prep_bpfilter(data,deg_filter+4,70,170,Fs); %High gamma band
        dataTrial_complex = hilbert(dataTrial_filtered);
        dataTrial_abs = abs(dataTrial_complex); %Getting the envelope of high gamma band
        tmpMat = prep_lpfilter(dataTrial_abs,deg_filter,lp_freq,Fs); %Filtering the envelope of data
        dataTrial_HG = cat(1,dataTrial_HG,zscore(tmpMat,0,1));
        sttimi = cat(1,sttimi,stim);
    end
    stim = sttimi;
    clear sttimi;
    
    ind = stim>=0; % remove values -2 and -1 from the labels which are unknown
    stim = stim(ind);
    dataTrial_HG = dataTrial_HG(ind,:);
    
%     dataTrial_HG = zscore(dataTrial_HG,0,1);
%     dataTrial_HG = dataTrial_HG(:,colMaxR2);
    clear dataTrial_filtered dataTrial_complex dataTrial_abs data flex classMean cue brain;
    dataTrial_HG = downsample(dataTrial_HG,50);
%     dataTrial_HG = zscore(dataTrial_HG,0,1);
    stim = downsample(stim,50);
    [numData,numCol] = size(dataTrial_HG);
    numRounds = 5;
    numData_10 = ceil(numData / numRounds);
    
    dataTrial_HG = cat(1,dataTrial_HG,zeros(numRounds*numData_10-numData,numCol));
    stim = cat(1,stim(:),zeros(numRounds*numData_10-numData,1));    
    dataTrial_HG = dataTrial_HG';
    stim = stim';
    
    disp('Partitioning...');
    dataTrial_HG_seq = reshape(dataTrial_HG,[numCol,numData_10,numRounds]);    
    dataTrial_HG_cell = mat2cell(dataTrial_HG_seq,numCol,numData_10,ones(1,numRounds));    
    dataTrial_HG_cell = permute(dataTrial_HG_cell,[1 3 2]);
    
    stim_seq = reshape(stim,[1,numData_10,numRounds]);
    stim_cell = mat2cell(stim_seq,1,numData_10,ones(1,numRounds));
    stim_cell = permute(stim_cell,[1 3 2]);
    clear stim stim_seq dataTrial_HG dataTrial_HG_seq;
    disp('Training...');
%     C = 0.05:0.05:1;
    C = 0.0;
%     numIt = [1 2 5 10 20 50 100];
    numIt = [50];
    lenC = numel(numIt);
    myAcc = zeros(numRounds,lenC);
    vecRounds = 1:numRounds;
    winSize = 0;
    modelT = 'crf';
    for elC = 1:lenC
        disp(['Training and testing CRF with numIt = ',num2str(numIt(elC)),'...']);
        paramHCRF = struct('modelType',modelT,'caption',upper(modelT),'optimizer','bfgs','regFactorL2',C,'regFactorL1',C,'windowSize',winSize,'rangeWeigths',[-2 2],'debugLevel',0);
        
        paramHCRF.nbHiddenStates = maxStates;
%         paramHCRF.normalizeWeights = 1;
        paramHCRF.maxIterations = numIt(elC);
        params.weightsInitType = 'RANDOM';
        for nTrain = 1:numRounds
            disp(['      ... with round ' num2str(nTrain) ' ...']);
            dataTrial_HG_test = dataTrial_HG_cell(vecRounds==nTrain);
            stim_test = stim_cell(vecRounds==nTrain);
            
            dataTrial_HG_train = dataTrial_HG_cell(vecRounds~=nTrain);
            stim_train = stim_cell(vecRounds~=nTrain);
            
            [modelCRF statsCRF] = trainCRF(dataTrial_HG_train, stim_train, paramHCRF);
            loglikeCRF = testCRF(modelCRF, dataTrial_HG_test, stim_test);
%             return
            loglikeCRF = cell2mat(loglikeCRF);
            [~,maxlogCRF] = max(loglikeCRF,[],1);
            maxlogCRF = maxlogCRF - 1;
            stim_test = cell2mat(stim_test);
            isEqualMaxlogStim = (maxlogCRF(:) == stim_test(:));
            myAcc(nTrain,elC) = mean(isEqualMaxlogStim);
        end
    end
    save(['data' filesep subject filesep subject '_' Type '_model_' upper(modelT) '_all_windowSize_' num2str(winSize) '_lpfreq_' num2str(lp_freq) '.mat'],'myAcc','C','dataTrial_HG_cell','stim_cell','numIt');
end