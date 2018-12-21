function [Y, partVec,condVec] = makePatterns(varargin)
% SArbuckle 10/2018
% EBerlot changes (Nov2018):
% - signal and noise separately
% - partVec and condVec outputs
% % housekeeping
G     = []; % condition covariance matrix
Sw    = []; % spatial covariance matrix
nPart = []; % number of partitions
nVox  = []; % number of voxels
signal= []; % signal
vararginoptions(varargin,{'G','Sw','nPart','nVox','signal'});
nCond = size(G,1);

% % simulate patterns with condition covariance G and spatial
% covariance SW at desired snr ratio.
% - make design matrix
X = kron(ones(nPart,1),eye(nCond));
% - simulate white data with zero mean
U = mvnrnd_exact(G,nVox);
if ~isempty(Sw)
    U = U * Sw^0.5; % Does not work right now
end
% - apply design to true patterns
Y = X*U*signal;
% add signal scaling
% - generate IID noise
% - scale noise by signal-to-noise signal-to-noise variance ratio
% - add noise to patterns
partVec = kron([1:nPart]',ones(nCond,1));            % Partitions
condVec = kron(ones(nPart,1),[1:nCond]');
end
