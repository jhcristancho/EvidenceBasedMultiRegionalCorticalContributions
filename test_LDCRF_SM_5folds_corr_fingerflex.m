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
% maxStates = 2:8;
maxStates = 2:5;
% maxStates = 0;
winMax = 2;
% winMax = 0;
delayMax = 0;
% delayMax = 0;
% Type = {'HG','LP','HG_LP'};
% Type = {'HG','LP'};
Type = {'HG'};
lp_freq = [8];
% lp_freq = [100];
fac_resample = [50]; %Resampling of 20 Hz
% fac_resample = [4]; %Resampling of 200 Hz
numIt = [100];
modelT = 'ldcrf';

% Type = 'LP';
% Type = 'HG_LP';

% lengthTrial = round(0.9*Fs); %Original length of trial: 0.9 seconds
% percentBaseline = 1.5*0.2*Fs / lengthTrial; %In percentage.
% It is the percentage of the total trial before the onset. The actual
% value of the baseline is 300 ms in the original length of trial.
% segment2chop = 0.5*0.2*Fs; %It is the lenght of the chopper for each extreme of a trial (see below)

for q=1:size(subjects,1)
    clearvars -except q subjects Fs maxStates lp_freq fac_resample Type winMax delayMax numIt modelT;
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
    nonSM = or(elec_regions > 4,elec_regions < 1);
    data(:,nonSM) =  []; %Only takes electrode in SM
    data = detrend(data); %Detrending
    data = ref_car(data); %Common Average Removal
    
    
    deg_filter = 4;
    %     deg_filter = 10;
    
    disp('Filtering...');
    lafreq = numel(lp_freq);
    
    for eltipo = 1:numel(Type)
        dataTrial_HG = [];
        sttimi = [];

        if strfind(Type{eltipo},'LP') > 0
            tmpMat = prep_bpfilter(data,deg_filter,0.05,lp_freq(lafreq),Fs); %Lowpass band
            dataTrial_HG = cat(1,dataTrial_HG,zscore(tmpMat,0,1));
            sttimi = cat(1,sttimi,stim);
        end

        if strfind(Type{eltipo},'HG') > 0
            dataTrial_filtered = prep_bpfilter(data,deg_filter+4,70,170,Fs); %High gamma band
            dataTrial_complex = hilbert(dataTrial_filtered);
            dataTrial_abs = abs(dataTrial_complex); %Getting the envelope of high gamma band
            tmpMat = prep_lpfilter(dataTrial_abs,deg_filter,lp_freq(lafreq),Fs); %Filtering the envelope of data
            dataTrial_HG = cat(1,dataTrial_HG,zscore(tmpMat,0,1));
            sttimi = cat(1,sttimi,stim);
        end

        ind = sttimi>=0; % remove values -2 and -1 from the labels which are unknown
        sttimi = sttimi(ind);
        dataTrial_HG = dataTrial_HG(ind,:);
        dataTrial_HG = downsample(dataTrial_HG,fac_resample(lafreq));
        sttimi = downsample(sttimi,fac_resample(lafreq));
        clear dataTrial_filtered dataTrial_complex dataTrial_abs flex classMean cue brain;
        %     dataTrial_HG = zscore(dataTrial_HG,0,1);
        %     dataTrial_HG = dataTrial_HG(:,colMaxR2);
        [numData,numCol] = size(dataTrial_HG);
        numRounds = 5;
        numData_10 = ceil(numData / numRounds);

        dataTrial_HG = cat(1,dataTrial_HG,zeros(numRounds*numData_10-numData,numCol));
        sttimi = cat(1,sttimi(:),zeros(numRounds*numData_10-numData,1));
        dataTrial_HG = dataTrial_HG';
        sttimi = sttimi';
        [numCol,numData] = size(dataTrial_HG);
        numData_10 = ceil(numData / numRounds);
        %     dataTrial_HG = zscore(dataTrial_HG,0,1);

        disp('Partitioning...');
        dataTrial_HG_seq = reshape(dataTrial_HG,[numCol,numData_10,numRounds]);
        dataTrial_HG_cell = mat2cell(dataTrial_HG_seq,numCol,numData_10,ones(1,numRounds));
        dataTrial_HG_cell = permute(dataTrial_HG_cell,[1 3 2]);

        stim_seq = reshape(sttimi,[1,numData_10,numRounds]);
        stim_cell = mat2cell(stim_seq,1,numData_10,ones(1,numRounds));
        stim_cell = permute(stim_cell,[1 3 2]);
        %                 clear stim_seq dataTrial_HG sttimi dataTrial_HG_seq;
        disp('Training...');
        %     C = 0.05:0.05:1;
        C = 0.0;
        lenC = numel(numIt);
        myAcc = zeros(numRounds,lenC+2);
        avgAcc = zeros(winMax+1,numel(maxStates),numRounds);
        vecRounds = 1:numRounds;
        winSize = winMax;
        elC = lenC;
        nDelay = delayMax;
        for nTrain = 1:numRounds
            disp(['For subject ' subject ' and round ' num2str(nTrain) ':']);
            %         paramHCRF.normalizeWeights = 1;
            tic;
            paramHCRF = struct('modelType',modelT,'caption',upper(modelT),'optimizer','bfgs','regFactorL2',C,'regFactorL1',C,'rangeWeigths',[-2 2],'debugLevel',0);
            paramHCRF.maxIterations = numIt(elC);
            params.weightsInitType = 'RANDOM';
            dataTrial_HG_train = dataTrial_HG_cell(vecRounds~=nTrain);
            stim_train = stim_cell(vecRounds~=nTrain);
            dataTrial_HG_test = dataTrial_HG_cell(vecRounds==nTrain);
            stim_test = stim_cell(vecRounds==nTrain);
            nRounds2 = numel(dataTrial_HG_train);
            tmpAcc = zeros(winMax+1,numel(maxStates),nRounds2);
            for elstate = 1:numel(maxStates)
                paramHCRF.nbHiddenStates = maxStates(elstate);
                for winSize = 0:winMax
                    disp(['    Training and testing with ' upper(modelT) ' model, ' num2str(maxStates(elstate)) ' hidden states, numIt = ',num2str(numIt(elC)),', ' Type{eltipo} ' parameters, windSize ' num2str(winSize) ', delay ' num2str(nDelay) ' and ' num2str(lp_freq(lafreq)) ' Hz lowpass freq...']);
                    paramHCRF.windowSize = winSize;
                    lolo = 1:nRounds2;
                    for elRound = 1:nRounds2
                        dataTrain_HG_train = dataTrial_HG_train(lolo~=elRound);
                        stimTrain_train = stim_train(lolo~=elRound);
                        dataTrain_HG_test = dataTrial_HG_train(lolo==elRound);
                        stimTrain_test = stim_train(lolo==elRound);

                        modelCRF = trainCRF(dataTrain_HG_train, stimTrain_train, paramHCRF);
                        loglikeCRF = testCRF(modelCRF, dataTrain_HG_test, stimTrain_test);
                        loglikeCRF = cell2mat(loglikeCRF);
                        [~,maxlogCRF] = max(loglikeCRF,[],1);
                        maxlogCRF = maxlogCRF - 1;
                        stimTrain_test = cell2mat(stimTrain_test);
                        isEqualMaxlogStim = (maxlogCRF(:) == stimTrain_test(:));
                        tmpAcc(winSize+1,elstate,elRound) = mean(isEqualMaxlogStim);
                    end
                end

            end
            save(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_SMreg_numIt' num2str(numIt(elC)) '_lpfreq_' num2str(lp_freq(lafreq)) '_tmpRound_' num2str(nTrain) '.mat'],'tmpAcc','dataTrial_HG_train','dataTrial_HG_test','stim_train','stim_test','winMax','maxStates');
            tmpMeanAcc = mean(tmpAcc,3);
            avgAcc(:,:,nTrain) = tmpMeanAcc;
            [tmpMax,iidx] = max(tmpMeanAcc,[],1);
            [~,iidy] = max(tmpMax);
            iidx = iidx(iidy);
            paramHCRF.nbHiddenStates = maxStates(iidy);
            paramHCRF.windowSize = iidx - 1;
            [modelCRF statsCRF] = trainCRF(dataTrial_HG_train, stim_train, paramHCRF);
            loglikeCRF = testCRF(modelCRF, dataTrial_HG_test, stim_test);
            loglikeCRF = cell2mat(loglikeCRF);
            [~,maxlogCRF] = max(loglikeCRF,[],1);
            maxlogCRF = maxlogCRF - 1;
            stim_test = cell2mat(stim_test);
            isEqualMaxlogStim = (maxlogCRF(:) == stim_test(:));
            myAcc(nTrain,lenC) = mean(isEqualMaxlogStim);
            myAcc(nTrain,lenC+1) = maxStates(iidy);
            myAcc(nTrain,lenC+2) = iidx - 1;
            toc;
            save(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_SMreg_numIt' num2str(numIt(elC)) '_lpfreq_' num2str(lp_freq(lafreq)) '_tmpRound_' num2str(nTrain) '.mat'],'tmpAcc','dataTrial_HG_train','dataTrial_HG_test','stim_train','stim_test','winMax','maxStates','myAcc','avgAcc');
        end

        save(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_SMreg_numIt' num2str(numIt(elC)) '_lpfreq_' num2str(lp_freq(lafreq)) '_V2_5.mat'],'myAcc','avgAcc','dataTrial_HG_cell','stim_cell');
        for nTrain = 1:numRounds
            delete(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_SMreg_numIt' num2str(numIt(elC)) '_lpfreq_' num2str(lp_freq(lafreq)) '_tmpRound_' num2str(nTrain) '.mat']);
        end
    end
end