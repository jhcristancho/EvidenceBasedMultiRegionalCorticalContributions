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

numSubjects = size(subjects,1);

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

numNatureData = 2;

numData = numNatureData*tw_len*nd_len*numRounds*numSubjects;

cell_conf = cell(numData,1);

mat_kappa = zeros(numData,1);
mat_acc = zeros(numData,1);
mat_subj = zeros(numData,1);
mat_twin = zeros(numData,1);
mat_nd = zeros(numData,1);

countdata = 1;
uniqueVec = 0:5;


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
    
    load(['data' filesep subject filesep subject '_dataTrial_HG_crf_res_corr.mat'],'metric_all_corr','metric_all','testCP_all','testCP_all_corr');
    
    for curtw = 1:tw_len
        winSize = round(TW(curtw)*Fs);
        for curnd = 1:nd_len
            sldSize = ND(curnd);
            for curtrial = 1:numRounds
                currAcc = metric_all(curtw,curnd,curtrial);
                cpData = testCP_all{curtw,curnd,curtrial};
                confSum = cpData.countingMatrix;  
                [fil,col] = size(confSum);
                if fil ~= col
                    confSum = cat(2,confSum,zeros(fil,1));
                end
                fufu = getkc(confSum);
                mat_kappa(countdata) = fufu;
                mat_acc(countdata) = currAcc;
                mat_subj(countdata) = q;
                mat_twin(countdata) = winSize;
                mat_nd(countdata) = sldSize;
                cell_conf{countdata} = 'No corr';
                countdata = countdata + 1;
                
                currAcc = metric_all_corr(curtw,curnd,curtrial);
                cpData = testCP_all_corr{curtw,curnd,curtrial};
                confSum = cpData.countingMatrix;  
                [fil,col] = size(confSum);
                if fil ~= col
                    confSum = cat(2,confSum,zeros(fil,1));
                end
                fufu = getkc(confSum);
                mat_kappa(countdata) = fufu;
                mat_acc(countdata) = currAcc;
                mat_subj(countdata) = q;
                mat_twin(countdata) = winSize;
                mat_nd(countdata) = sldSize;
                cell_conf{countdata} = 'With corr';
                countdata = countdata + 1;
                
            end
        end
    end
    
end

cell_group = {cell_conf,mat_twin,mat_subj};
cell_varnames = {'Presence of correlation','Window size','Subject'};
modelo = 'linear';
[anova_p_acc,anova_table_acc,anova_stats_acc,anova_terms_acc] = anovan(mat_acc,cell_group,'model',modelo,'varnames',cell_varnames,'display','on');
[anova_c1_acc,anova_m1_acc,anova_h1_acc,anova_gnames1_acc] = multcompare(anova_stats_acc,'dimension',1,'ctype','hsd','display','on');
set(gcf,'Color','white');
xlabel('Accuracy','FontSize',14);
set(gca,'FontSize',14);
set(findall(gca, 'Type', 'Line'),'LineWidth',2);