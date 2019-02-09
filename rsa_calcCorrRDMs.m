function rsa_corr = rsa_calcCorrRDMs(rdmSets,varargin)
% function rsa_corr = rsa_calcCorrRDMs(rdms)
% calculates correlation between sets of rdms
% based on upper triangular elements (excluding diagonal)
%
% INPUT:
% rdmSets:          structure with different rdm matrices sets
%
% VARARGIN:
% interceptFix:     whether or not intercept is fixed for correlation
%                   options - 0: not fixed (corr), 1: fixed (corrN)
%                   default: 0
%
% OUTPUT:
% rsa_corr:         estimated rsa_correlation between rdm sets


interceptFix = 0;
vararginoptions(varargin,{'interceptFix'});
numRDMs=size(rdmSets,2);
% string RDMs into vectors
vec = cell(1,numRDMs);
for st = 1:numRDMs
    mask = triu(true(size(rdmSets{st})),1);
    vec{st} = rdmSets{st}(mask);
end

% determine all possible pairs of rdms
indPair = indicatorMatrix('allpairs',1:numRDMs);
rsa_corr=zeros(1,size(indPair,1));
% make calculation for each pair
for p = 1:size(indPair,1)
    indRDM=find(indPair(p,:));
    % if no data - make nan
    if isempty(vec{indRDM(1)}) || isempty(vec{indRDM(2)})
        rsa_corr(p)=NaN;
    else
    switch interceptFix
        case 0
            rsa_corr(p)=corr(vec{indRDM(1)},vec{indRDM(2)});
        case 1
            rsa_corr(p)=corrN(vec{indRDM(1)},vec{indRDM(2)});
    end
    end
end
