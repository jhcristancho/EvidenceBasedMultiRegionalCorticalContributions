freq_vec = [5 10 20 50 100 200];
len_freq = length(freq_vec);
for fr = 1:len_freq
    freq = freq_vec(fr);
    eval(['call_sub_getFeat_HG_l07_d02_forLDCRF_' num2str(freq) 'Hz;']);
end