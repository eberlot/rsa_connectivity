function D = KLdivergence(S1,S2,m1,m2,varargin)
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
scale=[];
vararginoptions(varargin,{'scale'}); 

if isempty(scale)
    s = fminsearch(@(s) 0.5*(log(det(s*S2)/det(S1)) + trace(inv(s*S2)*S1) + (m2-m1)'*inv(s*S2)*(m2-m1) - size(S1,1)),1);
else
    s = scale;
end

D = 0.5*(log(abs(det(s*S2)/det(S1))) + trace(inv(s*S2)*S1) + (m2-m1)'*inv(s*S2)*(m2-m1) - size(S1,1));