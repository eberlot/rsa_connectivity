function S = rsa_calcdist(rdms,distType)
%function rsa_distcorr = rsa_calcdist(rdms,distType)
%calculates dsitances between sets of rdms
%provides a similarity matrix S
if nargin<2
    distType='Euclidean';
end

switch distType
    case 'distcorr'
        D = rsa_calcDistCorrRDMs(rdms);
    otherwise
        D = pdist(rdms,distType);
end

% Apply Gaussian similarity function
thres = quantile(D,0.05);
D     = squareform(D);
%S     = exp(-D.^2 ./ (2*thres^2));
S=D;
end