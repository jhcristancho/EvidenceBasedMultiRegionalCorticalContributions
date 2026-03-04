function [nodeMap,edgeMap,edgeLabelMap,labelMap,w] = train4HCRFwADJ(y_vec,x_cell,C_val,maxState,adjMat,x_edges_train)

unique_labels = unique(Y);
nLabels = length(unique_labels); %Number of possible values of y
nXFeats = size(trainData.data{1},2); %This is the size of each feature vector for each node with dimensionality R^d_node

nEdgeFeats = size(x_edges_train,1); %This is the size of each feature vector for each edge with dimensionality R^d_edge

nInstances = length(y_vec); %Number of elements for training data

Xnode = cell(1,nInstances); %Input features, where each feature instance Xi has dimensionality R^(nNodes*nFeaturesForX)
Xedge = cell(1,nInstances); %Edges features, where each feature instance Xe has dimensionality R^(nEdges*nFeaturesForXedges)
y=zeros(1,length(y_vec)); %Output labels by instance
edgeStruct = cell(1,nInstances); %Cell array of edgeStruct
for instance = 1:nInstances
    Xnode{instance} = x_cell{instance}; %Input features for training data. The conjugate occurs is because UGM uses Xnodes{nInstances} features of dimensionality R^(nNodes*nFeaturesForX)
    Xedge{instance} = x_edges_train{instance};
    y(instance)     = y_vec(instance); %Output labels for training data
    nNodes          = size(Xnode{instance},1); %Number of nodes by instance
    nStates = double(maxState*ones(nNodes,1)); %Number of states by node and instance
    edgeStruct{instance} = UGM_makeEdgeStruct(adjMat, nStates, useMex, maxIter);
end

%Creation of maps. The potential function is defined as follows:
%sum_j( dot(Xnode{j},w(nodeMap(s{j},f))) + w(labelMap(s{j},label) )
%+
%sum_{(j,k) with Edges}( dot({Xedge{j,k},w(edgeMap(s{j},s{k},f_edge))) + w(edgeLabelMap(s{j},s{k},label)) )

%Creation of nodeMap, according with Quattoni
nodeMap = zeros(maxState, nXFeats,'int32');
featNo = 1;
for s = 1:maxState
    for f = 1:nXFeats
        nodeMap(s, f) = featNo;
        featNo = featNo + 1;
    end
end

%Creation of edgeMap, according with Quattoni (this matrix should be empty, following Quattoni restrictions)
edgeMap = zeros(maxState, maxState, nEdgeFeats, 'int32');
for s1 = 1:maxState
    for s2 = 1:maxState
        for l = 1:nEdgeFeats
            edgeMap(s1, s2, l) = featNo;
            featNo = featNo + 1;
        end
    end
end

%Creation of edgeLabelMap, according with Quattoni
edgeLabelMap = zeros(maxState, maxState, nLabels, 'int32');
for s1 = 1:maxState
    for s2 = 1:maxState
        for l = 1:nLabels
            edgeLabelMap(s1, s2, l) = featNo;
            featNo = featNo + 1;
        end
    end
end

%Creation of labelMap, according with Quattoni
labelMap = zeros(maxState, nLabels, 'int32');
for s = 1:maxState
    for l = 1:nLabels
        labelMap(s, l) = featNo;
        featNo = featNo + 1;
    end
end


nParams = featNo-1;

for instance = 1:nInstances
    edgeStruct{instance}.useMex = 0;
end

w = rand(nParams, 1);

% Set up regularization parameters
lambda = C_val*ones(size(w));
reglaFunObj = @(w)penalizedL2(w,@UGM_myHCRF_NLL,lambda,Xnode,Xedge,y,nodeMap,edgeMap,labelMap,edgeLabelMap, edgeStruct,@UGM_Infer_Chain);

% LBFGS to find the weights
display(['Training on 1000 test sequences for ',num2str(numofFeaturesVec(setFeat)),' features, trial ',num2str(trialNo)]);
options.LS=0;
options.TolFun=1e-3;
options.TolX=1e-3;
options.Method='lbfgs';
options.Display='on';
options.MaxIter=400;
options.DerivativeCheck='off';

w = minFunc(reglaFunObj,w, options);