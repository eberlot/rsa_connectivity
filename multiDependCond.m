function [R2,r] = multiDependCond(regA,regB,partVec,condVec,varargin);
%function [R2,r] = multiDependCond(regA,regB,partVec,condVec,varargin);
% calculates the multivariate dependency of patterns in regA and regB
% 1) splits regA and regB patterns across runs into train and test sets
% 2) calculates a second moment for train and test
% 3) uses multiple regression (OLS) to map G_B from B_A
% 4) establishes the consistency of regA->regB using crossvalidation
%
%  INPUT:
%   regA(k x nVoxA) - voxel responses in regA
%   regB(k x nVoxB) - voxel responses in regB
%   partVec(k x 1)  - partition vector
%   condVec(k x 1)  - condition vector 
% VARARGIN:
%   'removeMean'   : remove the mean activation across conditions per voxel
%       'default '1'
% OUTPUT:
%   R2 - crossvalidated R2 
%   r  - correlation of predicted vs. real pattern in regB

removeMean = 1;
vararginoptions(varargin,{'removeMean'});

% check sizes of inputs
if size(regA,1)~=size(regB,1)
    error('Number of rows in regA and regB must be the same!\n');
end
if size(condVec,1)~=size(partVec,1)
    error('Size of condition and partition vector must be the same!\n');
end
if size(condVec,1)~=size(regA,1)
    error('Wrong size of inputs!\n');
end
part    = unique(partVec);
nPart   = numel(part);
nCond   = numel(unique(condVec));
nVoxA   = size(regA,2);
nVoxB   = size(regB,2);
% remove mean pattern per run
A = zeros(nCond,nVoxA,nPart);
B = zeros(nCond,nVoxB,nPart);
X = indicatorMatrix('identity_p',condVec);

% Estimate condition means within each run
for i=1:nPart
    Xa = X(partVec==part(i),:);
    Aa = regA(partVec==part(i),:);
    Ba = regB(partVec==part(i),:);
    A(:,:,i) = pinv(Xa)*Aa;
    B(:,:,i) = pinv(Xa)*Ba;
end;

if removeMean
    A=bsxfun(@minus,A,sum(A,1)/nCond);
    B=bsxfun(@minus,B,sum(B,1)/nCond);
end;

% reshape back
A_rsh = zeros(size(regA));
B_rsh = zeros(size(regB));
for i=1:nPart
    A_rsh(partVec==i,:) = A(:,:,i);
    B_rsh(partVec==i,:) = B(:,:,i);
end

% split into odd and even runs
A_even = A_rsh(mod(partVec,2)==0,:);
B_even = B_rsh(mod(partVec,2)==0,:);
A_odd  = A_rsh(mod(partVec,2)==1,:);
B_odd  = B_rsh(mod(partVec,2)==1,:);
pVec1  = partVec(mod(partVec,2)==0);
pVec2  = partVec(mod(partVec,2)==1);
cVec1  = condVec(mod(partVec,2)==0);
cVec2  = condVec(mod(partVec,2)==1);
% calculate distances for each partition

D_A1 = rsa.distanceLDC(A_even,pVec1,cVec1);
D_B1 = rsa.distanceLDC(B_even,pVec1,cVec1);
D_A2 = rsa.distanceLDC(A_odd,pVec2,cVec2);
D_B2 = rsa.distanceLDC(B_odd,pVec2,cVec2);

% estimate A_train->B_train mapping using OLS (for 1st partition)
beta = pinv(D_A1)*D_B1;
B_pred  = D_A2*beta;
res     = D_B2 - B_pred; % residuals unaccounted for
% evaluate prediction of regB (B_pred)
r    = corr(B_pred(:),D_B2(:));
R2   = 1-(trace(res*res')/trace(D_B2*D_B2')); % 1 - SSR/SST

end