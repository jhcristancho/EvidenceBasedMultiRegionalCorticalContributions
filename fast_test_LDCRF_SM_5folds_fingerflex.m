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
maxStates = [0,2:5];
% maxStates = 2:5;
% maxStates = 0;
winMax = 2;
% winMax = 0;
delayMax = 0;
% delayMax = 0;
Type = {'HG','LP','HG_LP'};
% Type = {'HG','LP'};
% Type = {'HG'};
lp_freq = [8];
% lp_freq = [100];
fac_resample = [50]; %Resampling of 20 Hz
% fac_resample = [4]; %Resampling of 200 Hz
numIt = [100];
% modelT = 'ldcrf';
modelT = {'crf', 'ldcrf'};
numRounds = 5;
numType = numel(Type);
numSubjects = size(subjects,1);
numModels = numel(modelT);

% Type = 'LP';
% Type = 'HG_LP';

% lengthTrial = round(0.9*Fs); %Original length of trial: 0.9 seconds
% percentBaseline = 1.5*0.2*Fs / lengthTrial; %In percentage.
% It is the percentage of the total trial before the onset. The actual
% value of the baseline is 300 ms in the original length of trial.
% segment2chop = 0.5*0.2*Fs; %It is the lenght of the chopper for each extreme of a trial (see below)

for q=1:numSubjects
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
    for eltipo = 1:numType
        for elmodelo = 1:numModels
            %                     disp(['Count of data: ' num2str(countdata)]);
            %                     load(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_nHid_' num2str(maxStates(elstate)) '_SMreg_windowSize_' num2str(winMax(winSize)) '_lpfreq_' num2str(lp_freq(lafreq)) '.mat']);
            nomfil = ['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT{elmodelo}) '_SMreg_numIt' num2str(numIt) '_lpfreq_' num2str(lp_freq) '_V2_5.mat'];
            %                         disp(nomfil);
            if exist(nomfil,'file')
%                 disp(nomfil);
                load(nomfil);
                tic;
                C = 0.0;
                paramHCRF = struct('modelType',modelT{elmodelo},'caption',upper(modelT{elmodelo}),'optimizer','bfgs','regFactorL2',C,'regFactorL1',C,'rangeWeigths',[-2 2],'debugLevel',0);
                paramHCRF.maxIterations = numIt(1);
                params.weightsInitType = 'RANDOM';
                numRounds = numel(dataTrial_HG_cell);
                vecRounds = 1:numRounds;
                myAcc_V2 = myAcc;
                testCP = cell(numRounds,1);
                for nTrain = 1:numRounds
                    disp(['For subject ' subject ' and round ' num2str(nTrain) ' and model ' modelT{elmodelo} ':'])
                    dataTrial_HG_train = dataTrial_HG_cell(vecRounds~=nTrain);
                    stim_train = stim_cell(vecRounds~=nTrain);
                    dataTrial_HG_test = dataTrial_HG_cell(vecRounds==nTrain);
                    stim_test = stim_cell(vecRounds==nTrain);
                    paramHCRF.nbHiddenStates = myAcc(nTrain, 2);
                    paramHCRF.windowSize = myAcc(nTrain, 3);
                    [modelCRF statsCRF] = trainCRF(dataTrial_HG_train, stim_train, paramHCRF);
                    loglikeCRF = testCRF(modelCRF, dataTrial_HG_test, stim_test);
                    loglikeCRF = cell2mat(loglikeCRF);
                    [~,maxlogCRF] = max(loglikeCRF,[],1);
                    maxlogCRF = maxlogCRF - 1;
                    stim_test = cell2mat(stim_test);
                    isEqualMaxlogStim = (maxlogCRF(:) == stim_test(:));
                    myAcc_V2(nTrain, 1) = mean(isEqualMaxlogStim);
                    testCP{nTrain} = classperf(stim_test, maxlogCRF,'Positive',1:5,'Negative',0);
                end
                toc;
                save(nomfil,'myAcc','dataTrial_HG_cell','stim_cell','myAcc_V2','avgAcc','testCP');
            end
        end
    end    
end