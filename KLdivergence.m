function [D,dim] = KLdivergence(S1,S2,m1,m2,varargin)
% function D = KLdivergence(S1,S2,m1,m2,s)
% calculates the KL divergence D between two sets of data (S1 || S2)
% INPUT:
%       - S1: covariance matrix from data 1
%       - S2: covariance matrix from data 2
%       - m1: mean of data 1
%       - m2: mean of data 1
% VARARGIN:
%       - scale:  scaling factor (from S2 to S1)
% OUTPUT:
%       - D: KL divergence metric
%
% if scaling factor is not provided, it is first estimated
% To handle rank-deficient S1 matrices, we first evaluate data only in the
% subspace spanned by S1 
scale=[];
vararginoptions(varargin,{'scale'}); 

% If no mean give - set to zero 
if (nargin<3 || isempty(m1))
    m1=zeros(size(S1,1),1); 
end; 
if (nargin<4 || isempty(m2))
    m2=zeros(size(S2,1),1); 
end; 

% Determine subspace that S1 lives in
[V,L]=eig(S1); 
[l,i]=sort(diag(L),1,'descend'); 
V=V(:,i); 
V=V(:,l>1e-10);  % Valuate only in the space that has any probability mass in S1

% Rotate S1 and S2 an mu's into the same subspace 
S1 = V'*S1*V; 
S2 = V'*S2*V; 
m1 = V'*m1; 
m2 = V'*m2; 

if(rank(S2)<rank(S1))
    D=inf; 
else 
    if isempty(scale)
        s = fminsearch(@(s) 0.5*(log(det(s*S2)/det(S1)) + trace((s*S2)\S1) + (m2-m1)'*((s*S2)\(m2-m1)) - size(S1,1)),1);
    else
        s = scale;
    end
    D = 0.5*(log(abs(det(s*S2)/det(S1))) + trace((s*S2)\S1) + (m2-m1)'*((s*S2)\(m2-m1)) - size(S1,1));
end;

dim = size(S1,1); % Dimensionality of the space we evaluate the KL in 
