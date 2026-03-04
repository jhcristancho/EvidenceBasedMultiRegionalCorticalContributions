function [pow_out,posi] = windslidepower(X,wins,slds,dimen)
if not(ismatrix(X));
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
% ncol = cols*(cols - 1)/2;
ncol = cols;
% disp(nseg);
% disp(ncol);
pow_out = zeros(nseg,ncol);
posi = zeros(nseg,1);

for elsegmento = 1:nseg
    lim_lo = (elsegmento - 1)*slds + 1;
    lim_hi = lim_lo + wins - 1;
%     disp(['Segment ' num2str(elsegmento) ' of ' num2str(nseg) '. lim_lo: ' num2str(lim_lo) ', lim_hi: ' num2str(lim_hi)]);
    tmptmp = X(lim_lo:lim_hi,:);
    tmptmp = tmptmp + 1e-6*randn(size(tmptmp));
    pow_triarray = abs(tmptmp).^2;
    pow_out(elsegmento,:) = sum(pow_triarray,1);
    posi(elsegmento) = lim_lo;
end
if dimen == 2
    pow_out = pow_out';
end