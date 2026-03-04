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
% maxStates = [0,2:8];
maxStates = [0,2:5];
winMax = 2;
% delayMax = 4;
delayMax = 0;
% winMax = 0:2:8;
Type = {'HG','LP','HG_LP'};
% Type = {'HG'};
lp_freq = [8];
fac_resample = [50];
termin = {'_numIt100'};
% termin = {'','_numIt100','_numIt200'};
% Type = 'LP';
% Type = 'HG_LP';


numMaxStates = numel(maxStates);
numWinMax = numel(winMax);
num_lp_freq = numel(lp_freq);
numType = numel(Type);
numSubjects = size(subjects,1);
numDelays = delayMax + 1;
numTermin = numel(termin);

avgAccWinMax = zeros(numMaxStates,numWinMax,numSubjects,numTermin+1);

cell_subs = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_type = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_h_st = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_nwin = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_lpfs = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_data = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_dlyd = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
cell_term = cell(numType*num_lp_freq*numSubjects*numMaxStates*numWinMax*numDelays*(numTermin+1),1);
countdata = 1;


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
    
    for lafreq = 1:num_lp_freq
        
        
        for eltipo = 1:numType
            for nDelay = 0:delayMax
                for nterm = 1:numTermin
                    %                     disp(termin{nterm})
                    for winSize = 1:numWinMax
                        
                        
                        for elstate = 1:numMaxStates
                            if maxStates(elstate) == 0
                                modelT = 'crf';
                            else
                                modelT = 'ldcrf';
                            end
                            
                            
                            %                     disp(['Count of data: ' num2str(countdata)]);
                            %                     load(['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_nHid_' num2str(maxStates(elstate)) '_SMreg_windowSize_' num2str(winMax(winSize)) '_lpfreq_' num2str(lp_freq(lafreq)) '.mat']);
                            nomfil = ['data' filesep subject filesep subject '_' Type{eltipo} '_model_' upper(modelT) '_nHid_' num2str(maxStates(elstate)) '_SMreg' termin{nterm} '_wDelay' num2str(nDelay) '_windowSize_' num2str(winMax(winSize)) '_lpfreq_' num2str(lp_freq(lafreq)) '_V2.mat'];
                            %                         disp(nomfil);
                            if exist(nomfil,'file')
                                load(nomfil);
                                if strcmp(termin{nterm},'_numIt200')
                                    tmpMean = mean(myAcc,1);
                                    avgAccWinMax(elstate,winSize,q,nterm) = tmpMean(1);
                                    cell_subs{countdata} = repmat({subject},size(myAcc,1),1);
                                    cell_data{countdata} = myAcc(:,1);
                                    cell_type{countdata} = repmat({Type(eltipo)},size(myAcc,1),1);
                                    cell_lpfs{countdata} = repmat([lp_freq(lafreq)],size(myAcc,1),1);
                                    cell_h_st{countdata} = repmat([maxStates(elstate)],size(myAcc,1),1);
                                    cell_nwin{countdata} = repmat([winMax(winSize)],size(myAcc,1),1);
                                    cell_dlyd{countdata} = repmat([nDelay],size(myAcc,1),1);
                                    cell_term{countdata} = repmat({'_numIt150'},size(myAcc,1),1);
                                    countdata = countdata + 1;
                                    
                                    avgAccWinMax(elstate,winSize,q,nterm+1) = tmpMean(2);
                                    cell_subs{countdata} = repmat({subject},size(myAcc,1),1);
                                    cell_data{countdata} = myAcc(:,2);
                                    cell_type{countdata} = repmat(Type(eltipo),size(myAcc,1),1);
                                    cell_lpfs{countdata} = repmat([lp_freq(lafreq)],size(myAcc,1),1);
                                    cell_h_st{countdata} = repmat([maxStates(elstate)],size(myAcc,1),1);
                                    cell_nwin{countdata} = repmat([winMax(winSize)],size(myAcc,1),1);
                                    cell_dlyd{countdata} = repmat([nDelay],size(myAcc,1),1);
                                    cell_term{countdata} = repmat({'_numIt200'},size(myAcc,1),1);
                                    countdata = countdata + 1;
                                    
                                else
                                    avgAccWinMax(elstate,winSize,q,nterm) = mean(myAcc);
                                    cell_subs{countdata} = repmat({subject},numel(myAcc),1);
                                    cell_data{countdata} = myAcc;
                                    cell_type{countdata} = repmat(Type(eltipo),numel(myAcc),1);
                                    cell_lpfs{countdata} = repmat([lp_freq(lafreq)],numel(myAcc),1);
                                    cell_h_st{countdata} = repmat([maxStates(elstate)],numel(myAcc),1);
                                    cell_nwin{countdata} = repmat([winMax(winSize)],numel(myAcc),1);
                                    cell_dlyd{countdata} = repmat([nDelay],numel(myAcc),1);
                                    if strcmp(termin{nterm},'')
                                        cell_term{countdata} = repmat({'_numIt50'},numel(myAcc),1);
                                    else
                                        cell_term{countdata} = repmat({termin{nterm}},numel(myAcc),1);
                                    end
                                    countdata = countdata + 1;
                                end
                                
                                
                                
                                
                            end
                            
                        end
                    end
                end
            end
        end
    end
end

cell_subs = vertcat( cell_subs{:} );
cell_data = vertcat( cell_data{:} );
cell_type = vertcat( cell_type{:} );
cell_lpfs = vertcat( cell_lpfs{:} );
cell_h_st = vertcat( cell_h_st{:} );
cell_nwin = vertcat( cell_nwin{:} );
cell_dlyd = vertcat( cell_dlyd{:} );
cell_term = vertcat( cell_term{:} );
cell_strs = strcat(repmat('Term',numel(cell_term),1),cell_term,repmat('_',numel(cell_term),1),repmat('windowSize',numel(cell_nwin),1),num2str(cell_nwin));
% [~, ix] = sort(cell_strs);
% cell_strs = cell_strs(ix);
% cell_subs = cell_subs(ix);
% cell_data = cell_data(ix);
% cell_type = cell_type(ix);
% cell_lpfs = cell_lpfs(ix);
% cell_strs = cell_strs(ix);
% cell_h_st = cell_h_st(ix);
% cell_nwin = cell_nwin(ix);
% cell_dlyd = cell_dlyd(ix);
% cell_term = cell_term(ix);

% cell_strs = num2cell(cell_strs,2);

cell_group = {cell_h_st,cell_type,cell_subs};
cell_varnames = {'Hidden States','Data Type','Subject'};
% return
[anova_p,anova_table,anova_stats,anova_terms] = anovan(cell_data,cell_group,'model','linear','varnames',cell_varnames);
[anova_c1,anova_m1,anova_h1,anova_gnames1] = multcompare(anova_stats,'dimension',1,'ctype','hsd');
figure;
[anova_c2,anova_m2,anova_h2,anova_gnames2] = multcompare(anova_stats,'dimension',2,'ctype','hsd');

save(['data' filesep 'compare_all_LDCRF_nHidden_DelayANDwindowSize_V2.mat'],'-regexp','^cell_','^anova_','^avgAcc');