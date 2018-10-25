function randData = rsa_conn_genData(numCond,numVox,numRun,varargin)
% function randData = rdm_conn_genData(mu,sigma,numCond,numVox,numRun)
% 
% generates random data based on inputs
% INPUTS:   
% - mu      : mean of distribution
% - sigma   : std
% - numCond : number of conditions
% - numVox  : number of voxels
% - numRun  : number of runs
%
% OUTPUS:
% randData  : resulting matrix with random data
%             size condVec x voxelsize 

% default values
dataType = 'rand'; 
mu = 1;
sigma = 0.2;
% options: rand or fromRDM (multivariate normal)

vararginoptions(varargin,{'dataType','signal','noise','mu','sigma','RDM'});

% construct mean and covariance for data generation from RDM
if ~strcmp(dataType,'rand')
    % make mu and sigma from RDM
    if exist('RDM','var')
        mu      = mean(RDM);
        sigma   = pcm_makePD(RDM);
    end
    % here check if pthe right size compared to number of conditions
    if exist('mu','var') && exist('sigma','var')
        % check if the right size
        if size(mu,2) ~= numCond || size(sigma,2) ~= numCond
            error('Provided mu and sigma do not correspond the number of conditions provided');
        end
    elseif exist('RDM','var')
    else
        error('Not providing the necessary inputs: need to provide either 1) sigma and mu or 2) an RDM matrix');
    end
end

% generate randData
randData=[];
for r=1:numRun
    switch dataType
        case 'rand'
            randRun=normrnd(mu,sigma,[numCond,numVox]);
        case 'fromRDM'      
            signalDist = @(x) norminv(x,0,1);  % Standard normal inverse for Signal generation 
            pSignal = unifrnd(0,1,numCond,numVox); 
            U       = signalDist(pSignal); 
            E       = (U*U'); 
            Z       = E^(-0.5)*U;   % Make random orthonormal vectors 
            A       = pcm_diagonalize(RDM); 
            trueU   = A*Z(1:size(A,2),:)*sqrt(numVox); 
            randRun = bsxfun(@times,trueU,sqrt(signal));   % Multiply by (voxel-specific) signal scaling factor 
        case 'fromRDM+noise'
            if ~exist('signal','var')
                error('signal level not defined');
            end
            if ~exist('noise','var')
                error('noise level not defined');
            end
            
            % prepare for generating noise / signal
            noiseDist = @(x) norminv(x,0,1);   % Standard normal inverse for Noise generation            
            trueU = mvnrnd(mu,sigma,numVox)'*signal;
            % Now add the random noise
            pNoise  = unifrnd(0,1,numCond,numVox);
            Noise   = bsxfun(@times,noiseDist(pNoise),sqrt(noise));
            randRun = trueU + Noise;
            
    end
    randData=[randData;randRun];
end
