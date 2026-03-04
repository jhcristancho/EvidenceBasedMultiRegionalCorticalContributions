% function call_subs_processing
% executes the entire signal processing, etc for each motor subject
clear all
close all
clc

delete crf_grad.mex* forward_backward_crf.mex* viterbi_crf.mex* crf_herding.mex* crf_herding_2nd_order.mex* hidden_crf_herding.mex* hidden_crf_herding_2nd_order.mex* viterbi_crf.mex* viterbi_crf_2nd_order.mex* viterbi_hidden_crf.mex* viterbi_hidden_crf_2nd_order.mex*

% if matlabpool('size') == 0
%     matlabpool('open');
% end
fac_resample = 50;

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
lengthTrial = round(0.9*Fs); %Original length of trial: 0.9 seconds
percentBaseline = 1.5*0.2*Fs / lengthTrial; %In percentage.
% It is the percentage of the total trial before the onset. The actual
% value of the baseline is 300 ms in the original length of trial.
segment2chop = 0.5*0.2*Fs; %It is the lenght of the chopper for each extreme of a trial (see below)
numIterLBFGS = 750;
numIterLBFGS_lambda = 200;
% numIterLBFGS_burn = 150; % Enable only if hidden perceptron

% maxStates = 2:6;
maxStates = 1;
% lambda = [0 .01 .1 1]; %Disable if hidden perceptron
lambda = 0;
modelT = 'crf';
% lambda = 0; %Enable if hidden perceptron
% eta_mat = [1e-4 2e-4 5e-4 1e-3 2e-3 5e-3]; %Enable if SGD
% eta_mat = [100 250 500 1000 2500 5000]; %Enable if hidden perceptron
% rho = [0 0.01 0.02 0.05 0.1 0.2 0.5]; %Enable if hidden perceptron
len_states = length(maxStates);
eta_mat = 1e-4; %Enable if not hidden perceptron
len_lambda = length(lambda);
len_eta = length(eta_mat);
rho = 0; %Enable if not hidden perceptron
len_rho = length(rho);

% batch_size = 1;
winMax = 0:2;
len_winmax = length(winMax);

TW = [0.5 2];
tw_len = numel(TW);
ND = 1;
nd_len = numel(ND);

numRounds = 5;
vecRounds = 1:numRounds;


%% cycle through subjects
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
    
    load(['data' filesep subject filesep subject '_dataTrial_HG_full_chop_forLDCRF_corrzone.mat'],'cell_trials_all','cell_labels_all','cell_correl_all','zonas');
    
    metric_all = zeros(tw_len,nd_len,numRounds);
    metric_all_corr = zeros(tw_len,nd_len,numRounds);
    testCP_all = cell(tw_len,nd_len,numRounds);
    testCP_all_corr = cell(tw_len,nd_len,numRounds);
    
%     tic;
    for curtw = 1:tw_len
        winSize = round(TW(curtw)*Fs);
        for curnd = 1:nd_len
            sldSize = ND(curnd);
            
            cell_trials_model = squeeze(cell_trials_all{curtw,curnd});
            cell_labels_model = squeeze(cell_labels_all{curtw,curnd});
            cell_correl_model = squeeze(cell_correl_all{curtw,curnd});
            for curtrial = 1:numRounds
                ind_train = vecRounds ~= curtrial;
                ind_test = vecRounds == curtrial;
                num_train = sum(double(ind_train));
                num_test = sum(double(ind_test));
                train_X = cell_trials_model(ind_train);
                test_X = cell_trials_model(ind_test);
                train_T = cell_labels_model(ind_train);
                test_T = cell_labels_model(ind_test);
                train_C = cell_correl_model(ind_train);
                test_C = cell_correl_model(ind_test);
                train_XC = cellfun(@(v,w) [v;w],train_X,train_C,'UniformOutput',0);
                test_XC = cellfun(@(v,w) [v;w],test_X,test_C,'UniformOutput',0);
                
                vecRounds2 = vecRounds(1:end-1);
                tmpAcc = zeros(len_states,len_lambda,len_eta,len_rho,len_winmax,vecRounds(end-1));
                tmpAcc2 = zeros(len_states,len_lambda,len_eta,len_rho,len_winmax,vecRounds(end-1));
                for posstates = 1:len_states
                    no_hidden = maxStates(posstates);
                    %Lambda: Not for hidden perceptron
                    for poslambda = 1:len_lambda
                        myLambda = lambda(poslambda);
                        for poseta = 1:len_eta
                            eta = eta_mat(poseta);
                            for posrho = 1:len_rho
                                rho_val = rho(posrho);
                                for lenwinsize = 1:len_winmax
                                    winvec = winMax(lenwinsize);                                    
                                    for postest = 1:num_train
                                        postrain = vecRounds2 ~= postest;
                                        pos_test = vecRounds2 == postest;
                                        ttrain_X = train_X(postrain);
                                        ttrain_T = train_T(postrain);
                                        ttrain_XC = train_XC(postrain);
                                        ttest_X = train_X(pos_test);
                                        ttest_T = train_T(pos_test);
                                        ttest_XC = train_XC(pos_test);
                                        clc;
                                        disp(['Subject ' num2str(q) ' of ' num2str(size(subjects,1)) ':']);
                                        disp(['Model training and testing with window size ' num2str(winSize) ' and slide size ' num2str(sldSize) '. Main trial: ' num2str(curtrial) '...']);
                                        disp(['For CRF: Lambda value: ' num2str(myLambda) ', eta value: ' num2str(eta) ', rho value: ' num2str(rho_val) ', subtrial number ' num2str(postest) ', adjacent times: ' num2str(winvec) '...']);
                                        disp('Choosing parameters for data without correlation...');
                                        paramHCRF = struct('modelType',modelT,'caption',upper(modelT),'optimizer','lbfgs','regFactorL2',myLambda,'regFactorL1',0,'rangeWeigths',[-2 2],'debugLevel',0);
                                        paramHCRF.maxIterations = numIterLBFGS_lambda;
                                        paramHCRF.weightsInitType = 'RANDOM';
%                                         paramHCRF.nbHiddenStates = no_hidden;
                                        paramHCRF.windowSize = winvec;
                                        modelCRF = trainCRF(ttrain_X, ttrain_T, paramHCRF);
                                        loglikeCRF = testCRF(modelCRF, ttest_X, ttest_T);
                                        loglikeCRF = cell2mat(loglikeCRF);
                                        [~,maxlogCRF] = max(loglikeCRF,[],1);
                                        maxlogCRF = maxlogCRF - 1;
                                        stimTrain_test = cell2mat(ttest_T);
                                        isEqualMaxlogStim = (maxlogCRF(:) == stimTrain_test(:));                                        
                                        tmpAcc(posstates,poslambda,poseta,posrho,lenwinsize,postest) = mean(isEqualMaxlogStim);
                                        disp(mean(isEqualMaxlogStim));
                                        disp('Choosing parameters for data with correlation...');
                                        %                                     [trash,model2] = hidden_crf_herding(ttrain_XC, ttrain_T, ttest_XC, ttest_T, 'drbm_continuous', no_hidden, true, eta, rho_val, numIterLBFGS_lambda, numIterLBFGS_burn);
                                        modelCRF2 = trainCRF(ttrain_XC, ttrain_T, paramHCRF);
                                        loglikeCRF = testCRF(modelCRF2, ttest_XC, ttest_T);
                                        loglikeCRF = cell2mat(loglikeCRF);
                                        [~,maxlogCRF] = max(loglikeCRF,[],1);
                                        maxlogCRF = maxlogCRF - 1;
                                        stimTrain_test = cell2mat(ttest_T);
                                        isEqualMaxlogStim = (maxlogCRF(:) == stimTrain_test(:));                                        
                                        tmpAcc2(posstates,poslambda,poseta,posrho,lenwinsize,postest) = mean(isEqualMaxlogStim);
                                        disp(mean(isEqualMaxlogStim));
                                        
                                    end
                                end
                            end
                        end
                    end
                end
                tmpAvg = mean(tmpAcc,ndims(tmpAcc));
                [max1,indStat] = max(tmpAvg,[],1);
                [max2,indlmbd] = max(max1,[],2);
                [max3,indeta] = max(max2,[],3);
                [max4,indrho] = max(max3,[],4);
                [max5,indwin] = max(max4,[],5);
                winvec = winMax(indwin);
                rhoval = rho(indrho(indwin));
                eta = eta_mat(indeta(indrho(indwin)));
                myLambda = lambda(indlmbd(indeta(indrho(indwin))));
                no_hidden = maxStates(indStat(indlmbd(indeta(indrho(indwin)))));
                %                 [trash,model] = hidden_crf_herding(train_X, train_T, test_X, test_T, 'drbm_continuous', no_hidden, true, eta, rho_val, numIterLBFGS, numIterLBFGS_burn);
                disp(['Testing data for round ' num2str(curtrial) ' without correlation...']);
                paramHCRF = struct('modelType',modelT,'caption',upper(modelT),'optimizer','lbfgs','regFactorL2',myLambda,'regFactorL1',0,'rangeWeigths',[-2 2],'debugLevel',0);
                paramHCRF.maxIterations = numIterLBFGS;
                paramHCRF.weightsInitType = 'RANDOM';
%                 paramHCRF.nbHiddenStates = no_hidden;
                paramHCRF.windowSize = winvec;
                modelCRF = trainCRF(train_X, train_T, paramHCRF);
                loglikeCRF = testCRF(modelCRF, test_X, test_T);
                loglikeCRF = cell2mat(loglikeCRF);
                [~,maxlogCRF] = max(loglikeCRF,[],1);
                maxlogCRF = maxlogCRF - 1;
                stimTrain_test = cell2mat(test_T);
                isEqualMaxlogStim = (maxlogCRF(:) == stimTrain_test(:));  
                fufu = unique(stimTrain_test);
                fua = (fufu <= 1);
                fub = (fufu > 1);
                
                metric_all(curtw,curnd,curtrial) = mean(isEqualMaxlogStim);
                if sum(fua) == 0 || sum(fub) == 0
                    testCP_all{curtw,curnd,curtrial} = classperf(stimTrain_test(:),maxlogCRF(:));
                else
                    testCP_all{curtw,curnd,curtrial} = classperf(stimTrain_test(:),maxlogCRF(:),'Positive',fufu(fub),'Negative',fufu(fua));
                end
                tmpAvg = mean(tmpAcc2,ndims(tmpAcc2));
                [max1,indStat] = max(tmpAvg,[],1);
                [max2,indlmbd] = max(max1,[],2);
                [max3,indeta] = max(max2,[],3);
                [max4,indrho] = max(max3,[],4);
                [max5,indwin] = max(max4,[],5);
                winvec2 = winMax(indwin);
                rhoval2 = rho(indrho(indwin));
                eta2 = eta_mat(indeta(indrho(indwin)));
                myLambda2 = lambda(indlmbd(indeta(indrho(indwin))));
                no_hidden2 = maxStates(indStat(indlmbd(indeta(indrho(indwin)))));
                
                disp(['Testing data for round ' num2str(curtrial) ' with correlation...']);
                paramHCRF = struct('modelType',modelT,'caption',upper(modelT),'optimizer','lbfgs','regFactorL2',myLambda2,'regFactorL1',0,'rangeWeigths',[-2 2],'debugLevel',0);
                paramHCRF.maxIterations = numIterLBFGS;
                paramHCRF.weightsInitType = 'RANDOM';
%                 paramHCRF.nbHiddenStates = no_hidden2;
                paramHCRF.windowSize = winvec2;
                modelCRF2 = trainCRF(train_XC, train_T, paramHCRF);
                loglikeCRF2 = testCRF(modelCRF2, test_XC, test_T);
                loglikeCRF2 = cell2mat(loglikeCRF2);
                [~,maxlogCRF2] = max(loglikeCRF2,[],1);
                maxlogCRF2 = maxlogCRF2 - 1;
                stimTrain_test = cell2mat(test_T);
                fufu = unique(stimTrain_test);
                fua = (fufu <= 1);
                fub = (fufu > 1);                
                isEqualMaxlogStim = (maxlogCRF2(:) == stimTrain_test(:));                
                metric_all_corr(curtw,curnd,curtrial) = mean(isEqualMaxlogStim);
                if sum(fua) == 0 || sum(fub) == 0
                    testCP_all_corr{curtw,curnd,curtrial} = classperf(stimTrain_test(:),maxlogCRF2(:));
                else
                    testCP_all_corr{curtw,curnd,curtrial} = classperf(stimTrain_test(:),maxlogCRF2(:),'Positive',fufu(fub),'Negative',fufu(fua));
                end

            end
        end
    end
%     toc;
    save(['data' filesep subject filesep subject '_dataTrial_HG_crf_res_corrzone.mat'],'metric_all_corr','metric_all','testCP_all','testCP_all_corr','zonas');
end