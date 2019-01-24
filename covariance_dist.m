function [V,sig,eps]=covariance_dist(data,partVec,condVec,varargin)
%function [V,sig,eps]=covariance_dist(data,partVec,condVec)
%calculates the covariance of distance estimates V, together with the
%signal component (sig) and noise component (eps)
%
% INPUT:
%       - data:     beta estimates (K x P; K:cond/run, P:voxels)
%       - partVec:  partition vector (K x 1)
%       - condVec:  condition vector (K x 1)
%
% OUTPUT:
%       - V:        covariance of distance estimates
%       - sig:      signal component 
%       - eps:      error component
%
% VARARGIN:
%       - distType: 'crossval' or 'ncv' (whether distances are
%                   calculated with crossvalidation or not) - changes V
%                   formula
%
distType = 'crossval'; % crossval or ncv
vararginoptions(varargin,'distType');

X = indicatorMatrix('identity_p',condVec); % per condition
C = indicatorMatrix('allpairs',unique(condVec)'); % condition pairs contrast
nVox = size(data,2);
nPart = numel(unique(partVec));

switch distType % correct denominator in the V equation
case 'crossval'
    den = nPart * (nPart-1);
case 'ncv'
    den = nPart * nPart;
end

% calculate mean pattern across runs
D = pinv(X)*data;
G = D*D';
sig = -0.5*C*G*C'/nVox; % signal component
%eps = cov(C*D);
S=0;
for i = 1:nPart
    S = S + (data(partVec==i,:)-D)*(data(partVec==i,:)-D)'/(nPart-1)*nVox;
end
eps = C*S*C'; % noise component
V = 4*(sig.*eps)/nPart + 2*(eps.^2)/den;
