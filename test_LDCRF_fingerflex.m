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
%     'cc';...
%     'bp';...
%     'zt';...
%     'jp';...
    'ht';...
%     'mv';...
%     'wc';...
%     'wm';...
%     'jc';...
    ];

%%
Fs = 1000; %Frequency sample
maxStates = 6;
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
    load(['data' filesep subject filesep subject '_dataTrial_HG_full_chop.mat']);
    clases = unique(labelTrial);
    numClases = numel(clases);
    classMean = zeros(size(dataTrial_HG,1),size(dataTrial_HG,2),numClases);
    for laClase = 1:numClases
        tmpMat = dataTrial_HG(:,:,labelTrial == clases(laClase));
        classMean(:,:,laClase) = mean(tmpMat,3);
    end
    matR2Clases = zeros(numClases,numClases,size(dataTrial_HG,2));
    vecR2Norm = zeros(1,size(dataTrial_HG,2));
    for laClase1 = 1:numClases
        mat1 = classMean(:,:,laClase1);
        for laClase2 = 1:numClases
            if laClase2 ~= laClase1
                mat2 = classMean(:,:,laClase2);
                vec_r2 = rsqu(mat1,mat2);
                matR2Clases(laClase1,laClase2,:) = vec_r2;
            end
        end        
    end
    for col = 1:size(dataTrial_HG,2)
        vecR2Norm(col) = norm(matR2Clases(:,:,col));
    end
    
    [~,colMaxR2] = max(vecR2Norm);
    clear dataTrial_HG labelTrial vec_r2 mat1 mat2 tmpMat vecR2Norm matR2Clases col laClase1 laClase2;
    
    load(['data' filesep subject filesep subject '_stim'],'stim'); %Everything starts here :)
    
    load(['data' filesep subject filesep subject '_fingerflex']);    
    % [numData,numCol] = size(data);    
    
    data = detrend(data); %Detrending
    data = ref_car(data'); %Common Average Removal 
    data = data';
    stim = stim + 1;
    
    disp('Filtering...');
    dataTrial_filtered = prep_bpfilter(data,10,70,170,Fs); %High gamma band
    dataTrial_complex = hilbert(dataTrial_filtered);
    dataTrial_abs = abs(dataTrial_complex); %Getting the envelope of high gamma band
    dataTrial_HG = prep_lpfilter(dataTrial_abs,10,20,Fs); %Filtering the envelope of data
    dataTrial_HG = dataTrial_HG(:,colMaxR2);
    clear dataTrial_filtered dataTrial_complex dataTrial_abs data flex classMean cue brain;
    dataTrial_HG = downsample(dataTrial_HG,round(Fs/20));
    stim = downsample(stim,round(Fs/20));
    [numData,numCol] = size(dataTrial_HG);
    
    numData_10 = ceil(numData / 10);
    
    dataTrial_HG = cat(1,dataTrial_HG,zeros(10*numData_10-numData,1));
    stim = cat(1,stim(:),zeros(10*numData_10-numData,1));    
    dataTrial_HG = dataTrial_HG';
    stim = stim';
    
    disp('Partitioning...');
    dataTrial_HG_seq = reshape(dataTrial_HG,[1,numData_10,10]);
    dataTrial_HG_cell = mat2cell(dataTrial_HG_seq,1,numData_10,ones(1,10));
    dataTrial_HG_cell = permute(dataTrial_HG_cell,[1 3 2]);
    stim_seq = reshape(stim,[1,numData_10,10]);
    stim_cell = mat2cell(stim_seq,1,numData_10,ones(1,10));
    stim_cell = permute(stim_cell,[1 3 2]);
    clear stim stim_seq dataTrial_HG dataTrial_HG_seq;
    disp('Training...');
%     C = 0.05:0.05:1;
    C = 0.0;
    lenC = numel(C);
    myAcc = zeros(10,lenC);
    for elC = 1:lenC
        disp(['Training and testing with C = ',num2str(C(elC)),'...']);
        paramLDCRF = struct('modelType','ldcrf','caption','LDCRF','optimizer','lbfgs','maxIterations',200,'windowSize',0,'rangeWeigths',[-1 1],'debugLevel',0);
        
        paramLDCRF.nbHiddenStates = maxStates;
        paramLDCRF.normalizeWeights = 1;
        paramLDCRF.regFactor = C(elC);
        for nTrain = 1:10
            dataTrial_HG_test = dataTrial_HG_cell(nTrain);
            stim_test = stim_cell(nTrain);
            if nTrain-1 < 1
                limInf = [];
            else
                limInf = 1:nTrain-1;
            end
            if nTrain+1 > length(dataTrial_HG_cell)
                limSup = [];
            else
                limSup = nTrain+1:length(dataTrial_HG_cell);
            end
            susu = [limInf,limSup];
            dataTrial_HG_train = dataTrial_HG_cell(susu);
            stim_train = stim_cell(susu);
            
            [modelLDCRF statsLDCRF] = trainLDCRF(dataTrial_HG_train, stim_train, paramLDCRF);
            loglikeLDCRF = testLDCRF(modelLDCRF, dataTrial_HG_test, stim_test);

            loglikeLDCRF = cell2mat(loglikeLDCRF);
            [~,maxlogLDCRF] = max(loglikeLDCRF,[],1);
            stim_test = cell2mat(stim_test);
            isEqualMaxlogStim = (maxlogLDCRF(:) == stim_test(:));
            myAcc(nTrain,elC) = sum(double(isEqualMaxlogStim)) / numel(isEqualMaxlogStim);
        end
    end
    save(['data' filesep subject filesep subject '_res_LDCRF_nStates_6_windowSize_0.mat'],'myAcc','C','dataTrial_HG_cell','stim_cell');
end