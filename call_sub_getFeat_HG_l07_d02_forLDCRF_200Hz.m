% function call_subs_processing
% executes the entire signal processing, etc for each motor subject
clearvars -except freq_vec len_freq fr freq;
close all
clc

% if matlabpool('size') == 0
%     matlabpool('open');
% end

%New frequency: 1000 / 5 = 200 Hz
fac_resample = 5;

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
% percentBaseline = 1.5*0.2*Fs / lengthTrial; %In percentage.
% It is the percentage of the total trial before the onset. The actual
% value of the baseline is 300 ms in the original length of trial.
% segment2chop = 0.5*0.2*Fs; %It is the lenght of the chopper for each extreme of a trial (see below)

% maxStates = 5:5:20;
TW = [2];
tw_len = numel(TW);
ND = [1];
nd_len = numel(ND);

numRounds = 5;
% vecRounds = 1:numRounds;

%% cycle through subjects
for q=1:size(subjects,1)
    subject=subjects(q,:);
    clc;
    disp(['Getting data for subject ' num2str(q) ' (' subject ') of ' num2str(size(subjects,1)) ' and frequency sample ' num2str(Fs/fac_resample) 'Hz...']);
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
    
    load(['data' filesep subject filesep subject '_fingerflex'],'data','elec_regions');
    
    nonSM = or(elec_regions > 4,elec_regions < 1);
    datamean = zeros(size(data,1),4);
%     disp(unique(elec_regions));
    for numdata = 1:4
        lila = mean(data(:,elec_regions == numdata),2);
        if ~isnan(mean(lila))
            datamean(:,numdata) = lila;
        end
    end
    zonas = 1:4;
    zonas(sum(datamean,1) == 0) = [];
    datamean(:,sum(datamean,1) == 0) = [];    
    data = detrend(data); %Detrending
    data = ref_car(data); %Common Average Removal
    data(:,nonSM) =  []; %Only takes electrode in SM
    stimi = stim;
    
    
    
    dataTrial_filtered = prep_bpfilter(data,8,70,170,Fs); %High gamma band
    dataTrial_complex = hilbert(dataTrial_filtered);
    dataTrial_abs = abs(dataTrial_complex); %Getting the envelope of high gamma band
    dataTrial_HG = prep_lpfilter(dataTrial_abs,8,20,Fs); %Filtering the envelope of data
    dataTrial_HG = zscore(dataTrial_HG,0,1); %Z-scoring. Data will depend on shape rather than actual values
%     dataTrial_mu = prep_bpfilter(data,10,8,12,Fs); %Alpha/mu band
%     dataTrial_beta = prep_bpfilter(data,10,15,25,Fs); %High gamma band
%     return
    
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
            [corr_out,posi] = windslidecorr(datamean,winSize,sldSize,1);
            
%             [pow_out,posi] = windslidepower(dataTrial_HG,winSize,sldSize,1);
            dataTrial_HG_post = dataTrial_HG(posi,:);
%             dataTrial_HG_post = pow_out(posi,:);
%             dataTrial_HG_post = zscore(dataTrial_HG_post,0,1);
            stimi_post = stimi(posi);            
            ind = stimi_post>=0; % remove values -2 and -1 from the labels which are unknown
            stimi_post = stimi_post(ind);            
            stimi_post = stimi_post + 1; %From 1 to 6, rather than 0 to 5
            dataTrial_HG_post = dataTrial_HG_post(ind,:);
            corr_out = corr_out(ind,:);
            
            dataTrial_HG_post = downsample(dataTrial_HG_post,fac_resample);
            stimi_post = downsample(stimi_post,fac_resample);
            corr_out = downsample(corr_out,fac_resample);
            
            [numData,numCol] = size(dataTrial_HG_post);
            [numDataCor,numColCor] = size(corr_out);
            numData_10 = ceil(numData / numRounds);
            
            dataTrial_HG_post = cat(1,dataTrial_HG_post,zeros(numRounds*numData_10-numData,numCol));
            stimi_post = cat(1,stimi_post(:),ones(numRounds*numData_10-numData,1)); %Here, I add ones rather than zeros, because of the classes
            corr_out = cat(1,corr_out,zeros(numRounds*numData_10-numData,numColCor));
            
            dataTrial_HG_post = dataTrial_HG_post';
            stimi_post = stimi_post';
            corr_out = corr_out';
            
            [numCol,numData] = size(dataTrial_HG_post);
            numData_10 = ceil(numData / numRounds);
            
            %     disp('Partitioning...');
            dataTrial_HG_seq = reshape(dataTrial_HG_post,[numCol,numData_10,numRounds]);
            dataTrial_HG_cell = mat2cell(dataTrial_HG_seq,numCol,numData_10,ones(1,numRounds));
            dataTrial_HG_cell = permute(dataTrial_HG_cell,[1 3 2]);
            %     dataTrial_HG_cell = dataTrial_HG_cell';
            stim_seq = reshape(stimi_post,[1,numData_10,numRounds]);
            stim_cell = mat2cell(stim_seq,1,numData_10,ones(1,numRounds));
            stim_cell = permute(stim_cell,[1 3 2]);
            %     stim_cell = stim_cell';
            corr_seq = reshape(corr_out,[numColCor,numData_10,numRounds]);
            corr_cell = mat2cell(corr_seq,numColCor,numData_10,ones(1,numRounds));
            corr_cell = permute(corr_cell,[1 3 2]);
            %     clear('dataTrial');
            cell_trials_all{curtw,curnd} = dataTrial_HG_cell;
            cell_labels_all{curtw,curnd} = stim_cell;
            cell_correl_all{curtw,curnd} = corr_cell;
        end
    end
    
    save(['data' filesep subject filesep subject '_dataTrial_HG_full_chop_forLDCRF_200Hz.mat'],'cell_trials_all','cell_labels_all','cell_correl_all','zonas');
end