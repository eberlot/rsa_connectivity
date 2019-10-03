Act = function create_DNNactivation3(DNNname,varargin)
%function create_DNNactivation(DNNname)
%creates activation files for DNN
% INPUT:    DNNname (vgg or resnet)

fileDir = '/Users/eberlot/Dropbox (Diedrichsenlab)/Data/rsa_connectivity';
nStim = 96; % number of stimuli
vararginoptions(varargin,{'fileDir'});
switch DNNname
    case 'vgg16'
        nLayer = 16;
    case 'resnet50'
        nLayer = 50;
    case 'alexnet_imagenet'
        nLayer = 8;
        nStim = 72;
     case 'alexnet'
        nLayer = 8;
end
act = cell(nLayer,1);
indx=1; % for image files
fprintf('Making activation file %s:\n',DNNname);
cd(fileDir);
d=dir('image*.mat');
for f=1:size(d)
    t=load(d(f).name);
    for n=1:nLayer
        if indx==1 % initialise variables
            act{n} = zeros(nStim,length(t.acti{n}(:)));
        end
        act{n}(indx,:) = t.acti{n}(:); % consider the dimensions: images, units, noise
    end
    fprintf('%d.',indx);
    indx=indx+1;
end
% save the new activation file
save(fullfile(fileDir,sprintf('%s_activations.mat',DNNname)),'act');
