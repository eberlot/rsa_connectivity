function rdms = calcRDMs_specificCorr(numCond,numRDMs,rdmCorr);
% function rdms = calcRDMs_specificCorr(numCond,rdmCorr);
% creates rdms with a prespecified correlation of off-diag elements
% 
% INPUTS:
%        numCond:   number of conditions - rows / columns in RDM
%        numRDMs:   number of RDM matrices to create
%        rdmCorr:   prespecified correlation of rdms
%
% OUTPUT:
%       rdms:       creates a structure containing RDM matrices


% calculate number of off-diagonal elements required
numElements = numCond*(numCond-1)/2;

% create random vectors with required size of elements and numRDMs
randV = randn(numElements,numRDMs);
randV = bsxfun(@minus,randV,mean(randV));

% do the Cholesky decomposition of the desired correlation matrix R
R = [1 rdmCorr; rdmCorr 1];
randV = randV * inv(chol(cov(randV)));
randV = randV * chol(R);

% extract vectors from modified matrix and create rdms
for s = 1:numRDMs
    rdms{s} = rsa_squareRDM(randV(:,s)');
end

end