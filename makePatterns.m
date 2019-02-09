function [Y, partVec,condVec] = makePatterns(G,varargin)
% function [Y, partVec,condVec] = makePatterns(varargin)
% makes patterns of beta values
% INPUT:
%           - G:        true second moment matrix (K x K)
% VARARGIN:
%           - nPart:    number of partitions (M; default 8)
%           - nVox:     number of voxels (P; default 100)
%           - signal:   true signal (scalar value; default 1)
%
% OUTPUT:
%           - Y: voxel  beta patterns (dimensions: (KxM) x P)
%           - partVec:  partition vector 
%           - condVec:  condition vector
%           
% EBerlot (Nov2018):
% % housekeeping
nPart  = 8; % number of partitions
nVox   = 1000; % number of voxels
signal = 1; % signal
vararginoptions(varargin,{'nPart','nVox','signal'});
nCond = size(G,1);
if isempty(G)
    error('Need to provide a second moment matrix (G) as input.');
end
% % simulate patterns with condition covariance G
% - make design matrix
X = kron(ones(nPart,1),eye(nCond));
% - simulate white data with zero mean
U = mvnrnd_exact(G,nVox);
% - apply design to true patterns + add signal scaling
Y = X*U*signal;
% create partition / condition vectors
partVec = kron((1:nPart)',ones(nCond,1));          
condVec = kron(ones(nPart,1),(1:nCond)');
end
