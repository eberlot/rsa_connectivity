function rsa_distcorr = rsa_calcDistCorrRDMs(rdms)
% function rsa_corr = rsa_calcCorrRDMs(rdms)
% calculates correlation between sets of rdms
% based on upper triangular elements (excluding diagonal)

numRDMs=size(rdms,2);

% first check if all rdms have the same size
for st = 1:numRDMs
    sizeRDM(st) = size(rdms{st},1);
end
% give an error if all RDMs not the same size
if ~all(sizeRDM == sizeRDM(1))
    error('all RDMs must be the same size');
else
    numCond = sizeRDM(1);
    if numCond<4
        error('distance correlation needs a minimum of 4 conditions');
    end
end


% centering matrix H
H = eye(numCond) - ones(numCond)/numCond;
for st = 1:numRDMs
    % calculate doubly centered distance matrices
    rdmsC{st} = H*rdms{st}*H';
    
    % make a modified version of centered rdm
    rdmsMod{st}  = zeros(size(rdmsC{st}));
    % 1) modify off-diagonal elements
    
    offDiag_new{st} = (numCond/(numCond-1))*(rdmsC{st}-(rdms{st}/(numCond)));
    rdmsMod{st}     = offDiag_new{st};
    
    % 2) modify diagonal elements
    meanRow     = mean(rdms{st},1);
    meanAll     = mean(mean(rdms{st}));
    diagNew{st} = (numCond/(numCond-1))*(meanRow-meanAll);
    % assign to new rdm
    rdmsMod{st}(eye(size(rdmsMod{st}))==1) = diagNew{st};
end

% determine all possible pairs of rdms
indPair = indicatorMatrix('allpairs',[1:numRDMs]);
% make calculation for each pair
for p = 1:size(indPair,1)
    ind=find(indPair(p,:));
    
    % Calculate covariance and variances
    dcov  = (sum(sum(offDiag_new{ind(1)}.*offDiag_new{ind(2)})) - (2/(numCond-2))*sum(diagNew{ind(1)}.*diagNew{ind(2)}))*(1/(numCond*(numCond-3)));
    
    dvar1 = (sum(sum(offDiag_new{ind(1)}.*offDiag_new{ind(1)})) - (2/(numCond-2))*sum(diagNew{ind(1)}.*diagNew{ind(1)}))*(1/(numCond*(numCond-3)));
    dvar2 = (sum(sum(offDiag_new{ind(2)}.*offDiag_new{ind(2)})) - (2/(numCond-2))*sum(diagNew{ind(2)}.*diagNew{ind(2)}))*(1/(numCond*(numCond-3)));
    
    % Calculate the distance correlation
    if dvar1 < 0 || dvar2 < 0     % check if variances larger than 0
        rsa_distcorr(p) = 0;     
    else                          % otherwise calculate correlation
        rsa_distcorr(p) = dcov/sqrt(dvar1*dvar2);
    end
end
