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
winMax = 2;
TW = [0.5 2];
tw_len = numel(TW);
ND = 1;
nd_len = numel(ND);

Type = {'HG'};
lp_freq = [8];
% lp_freq = [100];
fac_resample = [50]; %Resampling of 20 Hz
% fac_resample = [4]; %Resampling of 200 Hz
numIt = [100];
modelT = 'ldcrf';

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
            
            % Getting the correlations between channels
            cell_trials_all = cell(tw_len,nd_len);
            cell_labels_all = cell(tw_len,nd_len);
            cell_correl_all = cell(tw_len,nd_len);
            
            for curtw = 1:tw_len
                winSize = TW(curtw);
                for curnd = 1:nd_len
                    sldSize = ND(curnd);
                    disp(['Window Slide Correlation with window size ' num2str(winSize) ' and slide size ' num2str(sldSize) '...']);
                    winSize = round(TW(curtw)*Fs);
                    corr_out = windslidecorr(data,winSize,sldSize,1);
                    dataTrial_HG_post = dataTrial_HG(posi,:);
                end
            end
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
        
    end
end