function [R2] = multiDependVox(regA,regB,partVec,condVec,varargin)
%function [R2,r] = multiDependVox(regA,regB,partVec,condVec,varargin);
% calculates the multivariate dependency of voxel patterns in regA and regB
% 1) splits regA and regB patterns across runs
% 2) calculates a reduced pattern for regA using svd
% 3) uses multiple regression (OLS) to map regB from regA
% 4) establishes the consistency of regA->regB using crossvalidation
% based on Anzellotti (PLOSCompBio,2018)
%
%  INPUT:
%   regA(k x nVoxA) - voxel responses in regA
%   regB(k x nVoxB) - voxel responses in regB
%   partVec(k x 1)  - partition vector
%   condVec(k x 1)  - condition vector 
% VARARGIN:
%   'type'       : treat regA -> regB prediction as
%       'all'       - all voxA -> voxB
%       'reduceA'   - reduced regA dim -> voxB
%       'reduceAB'  - reduced regA dim -> reduced regB
%   'numDim'     : how many voxel dimensions to take
%       default '3'
%   'removeMean' : remove the mean voxel pattern across conditions
%       default '1' (mean removed)
% OUTPUT:
%   R2 - crossvalidated R2 
%   r  - correlation of predicted vs. real pattern in regB

numDim = 3;
type = 'all';
removeMean = 1;
vararginoptions(varargin,{'type','numDim','removeMean'});

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

% pre-allocate r and R2
r=zeros(nPart,1);
R2=zeros(nPart,1);
for i=1:nPart     
    % split regA into train and test
    switch type % take all voxels from regB or reduced dimensions
        case 'all'
            A_train = A_rsh(partVec~=i,:);
            A_test  = A_rsh(partVec==i,:);
            B_train = B_rsh(partVec~=i,:);
            B_test  = B_rsh(partVec==i,:);
        case 'reduceA'
            [u,s,v] = svd(A_rsh(partVec~=i,:));              % reduce regA train set using svd
            A_train = u*s(:,1:numDim);                         % reduced A_train dimension
            A_test  = A_rsh(partVec==i,:)*v(:,1:numDim);   % reduced A_test (using same svd dimensions)
            B_train = B_rsh(partVec~=i,:);
            B_test  = B_rsh(partVec==i,:);         
         case 'reduceAB'
            [u,s,v] = svd(A_rsh(partVec~=i,:));          
            A_train = u*s(:,1:numDim);                     
            A_test  = A_rsh(partVec==i,:)*v(:,1:numDim); 
            [u,s,v] = svd(B_rsh(partVec~=i,:));  
            B_train = u*s(:,1:numDim);   
            B_test  = B_rsh(partVec==i,:)*v(:,1:numDim);   
    end
    % estimate A_train->B_train mapping using OLS
    beta = pinv(A_train)*B_train;
    B_pred  = A_test*beta;
    res     = B_test - B_pred; % residuals unaccounted for
    % evaluate prediction of regB (B_pred)
    r(i)    = corr(B_pred(:),B_test(:));
    R2(i)   = 1-(trace(res*res')/trace(B_test*B_test')); % 1 - SSR/SST
end

%r = mean(r);
%R2 = mean(R2);
%r = 1-mean(r);
R2 = 1-mean(R2); % previous version - to make into distances

end