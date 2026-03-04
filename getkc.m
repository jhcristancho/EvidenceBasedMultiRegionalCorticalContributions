function [cohKap,seKap,prE,prA,prBA] = getkc(confMat)
%Input: confusion matrix
%Outputs:
%-Cohen's kappa index
%-Standard error
%-Accuracy
%-Probability by chance
%-Balanced accuracy
if size(confMat,1) ~= size(confMat,2)
    error('Input matrix must be square');
else
    N = sum(sum(confMat));
    prA = sum(diag(confMat)) / N; %Accuracy estimation
    numFC = size(confMat,1); %Number of row/colums
    prE = 0; %Probability by chance
    sumE = 0; %Sum of (n_i: + n_:i)
    for kk = 1:numFC
        prE = prE + sum(confMat(kk,:))*sum(confMat(:,kk)) / N^2;
        sumE = sumE + (sum(confMat(kk,:)) + sum(confMat(:,kk))) * (sum(confMat(kk,:))*sum(confMat(:,kk))) / N^3;
    end
    suMat = bsxfun(@rdivide,confMat,sum(confMat,1));
    diMat = diag(suMat);
    diMat = diMat(~isnan(diMat));
    prBA = mean(diMat);
    cohKap = (prA - prE) / (1 - prE);
    seKap = sqrt(prA + prE^2 - sumE)/(sqrt(N)*(1 - prE)); %Formula in 2016 paper
%     seKap = sqrt(prA*(1 - prA)/(N*(1-prE)^2)); %Formula given by Cohen
end