function rsa_distcorr = rsa_calcDistCorrRDMs(rdms)
% function rsa_corr = rsa_calcCorrRDMs(rdms)
% calculates correlation between sets of rdms
% based on upper triangular elements (excluding diagonal)

numRDMs=size(rdms,1);
numCond=size(rsa_squareRDM(rdms(1,:)),1);
if numCond<4
    error('distance correlation needs a minimum of 4 conditions');
end

% centering matrix H
H = eye(numCond) - ones(numCond)/numCond;
for st = 1:numRDMs
    % reconstruct the square matrix
    R{st}=rsa_squareRDM(rdms(st,:));
    % calculate doubly centered distance matrix
    R_cent{st} = H*R{st}*H';
    % make a modified version of centered rdm
    R_mod{st}  = zeros(size(R_cent{st}));
    % 1) modify off-diagonal elements
    offDiag_new{st} = (numCond/(numCond-1))*(R_cent{st}-(R{st}/(numCond)));
    R_mod{st}     = offDiag_new{st};
    % 2) modify diagonal elements
    meanRow     = mean(R{st},1);
    meanAll     = mean(mean(R{st}));
    diagNew{st} = (numCond/(numCond-1))*(meanRow-meanAll);
    % assign to new rdm
    R_mod{st}(eye(size(R_mod{st}))==1) = diagNew{st};
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
        rsa_distcorr(p) = 1-dcov/sqrt(dvar1*dvar2);
    end
end
