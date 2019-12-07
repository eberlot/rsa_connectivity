function act_noise = addNoise(act,varargin)
% function act_noise = addNoise(act)
% adding noise to activations
% INPUT:    - act: cell with activations per layer
% VARARGIN: - sigma: amount of noise
%           - normAct: whether to normalise activation patterns
% OUTPUT:   - act_noise: activations with added noise level       
% ---------
sigma       = 1; % amount of noise added - usually varied 0.25:0.25:3
normAct     = 1; % whether to normalise activation of units or not
vararginoptions(varargin,{'sigma','normAct'});
nLayer      = size(act,1);
nTrials     = size(act{1},1);
act_noise   = cell(size(act));
for i=1:nLayer
    if normAct      
        act{i}=bsxfun(@minus,act{i},mean(act{i},1));  % first here remove the mean activation
        act{i}=act{i}./max(max(act{i}));
    end
    % now add noise
    nVox = size(act{i},2);
    Z = normrnd(0,1,nTrials,nVox);
    Z = Z./max(max(Z));
    act_noise{i} = act{i} + sigma*Z;
    act_noise{i} = act_noise{i}./max(max(act_noise{i}));
end
