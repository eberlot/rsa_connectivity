function [R2,r] = multiDepend(regA,regB,partVec,condVec,varargin);
%function [R2,r] = multiDepend(regA,regB,partVec,condVec,varargin);
% calculates the multivariate dependency of patterns in regA and regB
% 1) splits regA and regB patterns across runs
% 2) calculates a reduced pattern for regA using svd
% 3) uses multiple regression (OLS) to map regB from regA
% 4) establishes the consistency of regA->regB using crossvalidation
% based on Anzellotti(PLOS Bio, 2018)
%
%  INPUT:
%   regA(k x nVoxA) - voxel responses in regA
%   regB(k x nVoxB) - voxel responses in regB
%   partVec(k x 1)  - partition vector
%   condVec(k x 1)  - condition vector 
% VARARGIN:
%   'type'      : treat regA -> regB prediction as
%       'all'       - all voxA -> voxB
%       'reduceA'   - reduced regA dim -> voxB
%       'reduceAB'  - reduced regA dim -> reduced regB
%   'numDim'    : how many voxel dimensions to take
%       default 3
% OUTPUT:
%   R2 - crossvalidated R2 
%   r  - correlation of predicted vs. real pattern in regB
numDim = 3;
type = 'all';
vararginoptions(varargin,{'type','numDim'});

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

nPart   = numel(unique(partVec));
for i=1:nPart     
    % split regA into train and test
    [u,s,v] = svd(regA(partVec~=i,:));          % reduce regA train set using svd 
    A_train = u*s(:,[1:3]);                     % reduced A_train dimension
    A_test  = regA(partVec==i,:)*v(:,[1:3]);    % reduced A_test (using same svd dimensions)  
    
    switch type % take all voxels from regB or reduced dimensions
        case 'all'
            A_train = regA(partVec~=i,:);
            A_test  = regA(partVec==i,:);
            B_train = regB(partVec~=i,:);
            B_test  = regB(partVec==i,:);
        case 'reduceA'
            [u,s,v] = svd(regA(partVec~=i,:));              % reduce regA train set using svd
            A_train = u*s(:,[1:numDim]);                         % reduced A_train dimension
            A_test  = regA(partVec==i,:)*v(:,[1:numDim]);   % reduced A_test (using same svd dimensions)
            B_train = regB(partVec~=i,:);
            B_test  = regB(partVec==i,:);         
         case 'reduceAB'
            [u,s,v] = svd(regA(partVec~=i,:));          
            A_train = u*s(:,[1:3]);                     
            A_test  = regA(partVec==i,:)*v(:,[1:numDim]); 
            [u,s,v] = svd(regB(partVec~=i,:));  
            B_train = u*s(:,[1:3]);   
            B_test  = regB(partVec==i,:)*v(:,[1:numDim]);   
    end
    % estimate A_train->B_train mapping using OLS
    beta = pinv(A_train)*B_train;
    B_pred  = A_test*beta;
    res     = B_test - B_pred; % residuals unaccounted for
    % evaluate prediction of regB (B_pred)
    r(i)    = mean(diag(corr(B_pred,B_test)));
    R2(i)   = 1-(trace(res*res')/trace(B_test*B_test')); % 1 - SSR/SST
end

r = mean(r);
R2 = mean(R2);

end