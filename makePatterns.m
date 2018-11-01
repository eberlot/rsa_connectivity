function [Y partVec condVec] = makePatterns(varargin)
% SArbuckle 10/2018
% % housekeeping
G     = []; % condition covariance matrix
Sw    = []; % spatial covariance matrix
nPart = []; % number of partitions
nVox  = []; % number of voxels
snr   = []; % signal-to-noise variance ratio (scale noise accordingly)
vararginoptions(varargin,{'G','Sw','nPart','nVox','snr'});
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

% - generate IID noise
% - scale noise by signal-to-noise signal-to-noise variance ratio
% - add noise to patterns
E = unifrnd(0,1,nCond*nPart,nVox);
%E = mvnrnd(zeros(nCond*nPart,1),eye(nCond*nPart),nVox*nCond)';
E = bsxfun(@times,E,sqrt(1/snr));
Y = Y + E;
partVec = kron([1:nPart]',ones(nCond,1));            % Partitions
condVec = kron(ones(nPart,1),[1:nCond]');
end
    