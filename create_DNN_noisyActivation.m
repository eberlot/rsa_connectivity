function create_DNN_noisyActivation(DNNname,baseDir,noiseLevel,numSim)
%function create_DNN_noisyActivation(DNNname)
%creates activation files for DNN
% INPUT:    - DNNname (vgg or resnet)
%           - baseDir ('/Users/eberlot/Dropbox(Diedrichsenlab)/Data/rsa_connectivity')
%           - noiseLevel: amount of noise (0.2)
%           - numSim: 100 (searches for files 1-100)
% ---------------------------------------------------
nStim = 96; % number of stimuli
switch DNNname
    case 'alexnet'
        nLayer = 8;
end

for stim = 1:nStim
    act = cell(nLayer,1);
    fprintf('\nStimulus %d:\n',stim);
    for n=1:numSim % number of simulations
        simDir = fullfile(baseDir,DNNname,sprintf('noiseActivation_%1.1f',noiseLevel));
        cd(fullfile(simDir,sprintf('noiseActivation_%d',n)));
        d=dir(sprintf('image%02d*.mat',stim));
        t=load(d.name);
        for l=1:nLayer
            if n==1 % initialise variables
                act{l} = zeros(numSim,length(t.acti{l}(:)));
            end
            act{l}(n,:) = t.acti{l}(:); % consider the dimensions: images, units, noise
        end
        fprintf('%d.',n);
    end
    % save the new activation file - per stimulus
    save(fullfile(baseDir,DNNname,sprintf('noiseActivation_%1.1f',noiseLevel),sprintf('%s_noiseActivations_%1.1f_stim-%d.mat',DNNname,noiseLevel,stim)),'act','-v7.3');
end


