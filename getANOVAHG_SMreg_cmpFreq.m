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
    'bp';...
    'cc';...
    'ht';...
    'jc';...
    'jp';...
    'mv';...
    'wc';...
    'wm';...
    'zt';...
    ];

%%
Fs = 1000; %Frequency sample
% Type = {'HG','LP','HG_LP'};
Type = {'HG'};
lp_freq = 8;
numIt = 100;
fac_resample = 50;
termin = {'_numIt100'};
modelT = {'crf', 'ldcrf'};
version_data = {'2_5','3'};
freq_data = {'8 Hz', '20 Hz'};
% termin = {'','_numIt100','_numIt200'};
% Type = 'LP';
% Type = 'HG_LP';
numRounds = 5;
numType = numel(Type);
numSubjects = size(subjects,1);
numTermin = numel(termin);
numModels = numel(modelT);
numVersion = numel(version_data);

cell_subs = cell(numType*numModels*numSubjects*numVersion,1);
cell_mode = cell(numType*numModels*numSubjects*numVersion,1);
cell_data = cell(numType*numModels*numSubjects*numVersion,1);
cell_conf = cell(numType*numModels*numSubjects*numVersion,1);
cell_kapa = cell(numType*numModels*numSubjects*numVersion,1);
cell_freq = cell(numType*numModels*numSubjects*numVersion,1);
cell_bacc = cell(numType*numModels*numSubjects*numVersion,1);
cell_mode2 = cell(numType*numModels*numSubjects*numVersion,1);


countdata = 1;
uniqueVec = 0:5;

matMeanAvg = zeros(numSubjects,numModels);
matKapaSub = zeros(numSubjects,numModels);
matKapaSTD = zeros(numSubjects,numModels);
matBalance = zeros(numSubjects,numModels);

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

    for elmodelo = 1:numModels
        for laversion = 1:numVersion
            %                     disp(['Count of data: ' num2str(countdata)]);
            %                     load(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_nHid_' num2str(maxStates(elstate)) '_SMreg_windowSize_' num2str(winMax(winSize)) '_lpfreq_' num2str(lp_freq(lafreq)) '.mat']);
            if ~isempty(strfind(modelT{elmodelo},'crf'))
                nomfil = ['data' filesep subject filesep subject '_' Type{1} '_model_' upper(modelT{elmodelo}) '_SMreg_numIt' num2str(numIt) '_lpfreq_' num2str(lp_freq) '_V'  version_data{laversion} '.mat'];

            else
                nomfil = ['data' filesep subject filesep subject '_' upper(modelT{elmodelo}) '_SMreg_lpfreq_' num2str(lp_freq) '_V' version_data{laversion} '.mat'];
            end

            if exist(nomfil,'file')
                disp(nomfil);
                load(nomfil);
                confMat = cell(numel(testCP),1);
                if isempty(strfind(modelT{elmodelo},'crf'))
                    myAcc_V2 = avgAcc;
                else
                    myAcc_V2 = myAcc_V2(:,1:3);
                end
                finV2 = size(myAcc_V2,2);
                myAcc_V2 = cat(2,myAcc_V2,zeros(size(myAcc_V2,1),5));
                for kk = 1: numel(confMat)
                    confMat{kk} = testCP{kk}.CountingMatrix;
                    A = unique(stim_cell{kk});

                    if ~isequal(A,uniqueVec)
                        [r c] = size(confMat{kk});
                        [~,eleq] = intersect(uniqueVec,A);
                        newConf = zeros(length(uniqueVec)+1,length(uniqueVec));
                        eleq1 = [eleq;length(uniqueVec)+1];
                        newConf(eleq1,eleq) = confMat{kk};
                        confMat{kk} = newConf;
                    end
                    [myAcc_V2(kk,finV2+1),myAcc_V2(kk,finV2+2),myAcc_V2(kk,finV2+3),myAcc_V2(kk,finV2+4),myAcc_V2(kk,finV2+5)] = getkc(confMat{kk}(1:end-1,:));
                end
                confSum = sum(cell2mat(reshape(confMat, [1 1 numel(confMat)])),3);
                confSum = cat(2,confSum,zeros(size(confSum,1),1));
                save(nomfil,'myAcc','avgAcc','dataTrial_HG_cell','stim_cell','testCP','myAcc_V2');
                cell_conf{countdata} = confSum;
                cell_subs{countdata} = repmat({subject},size(myAcc_V2,1),1);
                cell_data{countdata} = myAcc_V2(:,1);
                fufu = mean(myAcc_V2(:,1),1);
                matMeanAvg(q,elmodelo) = fufu;
                cell_kapa{countdata} = myAcc_V2(:,finV2+1);
                cell_bacc{countdata} = myAcc_V2(:,finV2+5);
                cell_freq{countdata} = repmat(freq_data(laversion),size(myAcc_V2,1),1);
                cell_mode{countdata} = repmat(modelT(elmodelo),size(myAcc_V2,1),1);
                cell_mode2{countdata} = modelT{elmodelo};
                [fufu, lala, ~, ~, momo] = getkc(confSum);
                matKapaSub(q,elmodelo) = fufu;
                matKapaSTD(q,elmodelo) = lala;
                matBalance(q,elmodelo) = momo;
                countdata = countdata + 1;
            end
        end
    end

end

% kappa_byType = zeros(1,numType*numModels);
% STD_kappa_byType = zeros(1,numType*numModels);
% cell_conf2 = cell(1,numType*numModels);
% cell_conMat = cell(1,numType*numModels);
% countdata = 1;
% for elmodelo = 1:numModels
%     for eltipo = 1:numType
%         %         celLabel{countdata} = [upper(modelT{elmodelo}) ' ' Type{eltipo}];
% 
%         yogi = ismember(cell_mode2,modelT(elmodelo));
%         lasMatrices = cell_conf(yogi);
%         confSum2 = sum(cell2mat(reshape(lasMatrices, [1 1 numel(lasMatrices)])),3);
%         cell_conf2{countdata} = confSum2;
%         confMat = bsxfun(@rdivide, confSum2, sum(confSum2,1));
%         cell_conMat{countdata} = confMat;
%         %         confSum2 = cat(2,confSum2,zeros(size(confSum2,1),1));
%         [kappa_byType(countdata),STD_kappa_byType(countdata)] = getkc(confSum2);
%         countdata = countdata + 1;
%     end
% end


cell_subs = vertcat( cell_subs{:} );
cell_data = vertcat( cell_data{:} );
cell_mode = vertcat( cell_mode{:} );
cell_kapa = vertcat( cell_kapa{:} );
cell_freq = vertcat( cell_freq{:} );

cell_group = {cell_freq,cell_mode,cell_subs};
cell_varnames = {'Sample Frequency','Classifier','Subject'};
% return
[anova_p_acc,anova_table_acc,anova_stats_acc,anova_terms_acc] = anovan(cell_data,cell_group,'model','linear','varnames',cell_varnames,'display','on');
figure;
[anova_c1_acc,anova_m1_acc,anova_h1_acc,anova_gnames1_acc] = multcompare(anova_stats_acc,'dimension',1,'ctype','hsd','display','on');
figure;
[anova_c2_acc,anova_m2_acc,anova_h2_acc,anova_gnames2_acc] = multcompare(anova_stats_acc,'dimension',2,'ctype','hsd','display','on');

figure;
[anova_p_kappa,anova_table_kappa,anova_stats_kappa,anova_terms_kappa] = anovan(cell_kapa,cell_group,'model','linear','varnames',cell_varnames,'display','on');
figure;
[anova_c1_kappa,anova_m1_kappa,anova_h1_kappa,anova_gnames1_kappa] = multcompare(anova_stats_kappa,'dimension',1,'ctype','hsd','display','on');
figure;
[anova_c2_kappa,anova_m2_kappa,anova_h2_kappa,anova_gnames2_kappa] = multcompare(anova_stats_kappa,'dimension',2,'ctype','hsd','display','on');

save(['data' filesep 'compare_all_Algorithm_V3.mat'],'-regexp','^cell_','^anova_','^avgAcc','^mat');