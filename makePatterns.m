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
noise = []; % noise
vararginoptions(varargin,{'G','Sw','nPart','nVox','signal','noise'});
nCond = size(G,1);
    
% % simulate patterns with condition covariance G and spatial
% covariance SW at desired snr ratio. 
% - make design matrix
X = kron(ones(nPart,1),eye(nCond));
% - simulate white data with zero mean
I = eye(nCond);
Z = mvnrnd(zeros(nCond,1),I,nVox);
Z = Z';
E = (Z*Z'); 
% - ensure white patterns are random & orthonormal with variance = 1
Z = E^(-0.5)*Z;   
% - apply condition covariances
A = pcm_diagonalize(G);              % safe decomposition of G into A*A'
U = A * Z(1:size(A,2),:)*sqrt(nVox); % enforce condition covariances
% - apply spatial kernel (if applicable)
if isempty(Sw)
    Y = U;
else
    Y = U * Sw^0.5;
end
% - apply design to true patterns
Y = X*Y;
% add signal scaling
Y = bsxfun(@times,Y,signal);
% - generate IID noise
% - scale noise by signal-to-noise signal-to-noise variance ratio
% - add noise to patterns
E = unifrnd(0,1,nCond*nPart,nVox);
% add noise scaling
E = bsxfun(@times,E,noise);
Y = Y + E;
partVec = kron([1:nPart]',ones(nCond,1));            % Partitions
condVec = kron(ones(nPart,1),[1:nCond]');
end
    