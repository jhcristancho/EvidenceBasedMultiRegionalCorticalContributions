function [corr_out,posi] = windslidecorr(X,wins,slds,dimen)
if not(ismatrix(X))
    error('Number of dimensions exceed 2.');
end
if nargin < 3
    error('Minimum number of inputs must be 3.');
end
if nargin == 3
    dimen = 1;
end
if dimen == 2 || isrow(X)
    X = X';
end
[rows,cols] = size(X); %Sliding window is applied in first dimension
% disp(rows);
nseg = floor((rows - wins)/slds) + 1;
ncol = cols*(cols - 1)/2;
% disp(nseg);
% disp(ncol);
corr_out = zeros(nseg,ncol);
posi = zeros(nseg,1);

for elsegmento = 1:nseg
    lim_lo = (elsegmento - 1)*slds + 1;
    lim_hi = lim_lo + wins - 1;
%     disp(['Segment ' num2str(elsegmento) ' of ' num2str(nseg) '. lim_lo: ' num2str(lim_lo) ', lim_hi: ' num2str(lim_hi)]);
    tmptmp = X(lim_lo:lim_hi,:);
    tmptmp = tmptmp + 1e-6*randn(size(tmptmp));
    tmp_cor = corr(tmptmp);
    t_l = tril(tmp_cor, -1);
%     disp(size(t_l));
    t_la = t_l(:)';
%     disp(numel(t_la));
    t_la(t_la == 0) = [];
%     disp(numel(t_la));
    corr_out(elsegmento,:) = t_la;
    posi(elsegmento) = lim_lo;
end
if dimen == 2
    corr_out = corr_out';
end