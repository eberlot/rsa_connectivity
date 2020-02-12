function create_DNN_trueActivation(DNNname,fileDir)
%function create_DNN_trueActivation(DNNname)
%creates activation files for DNN
% INPUT:    DNNname (vgg or resnet)
%           fileDir ('/Users/eberlot/Dropbox
%           (Diedrichsenlab)/Data/rsa_connectivity')

nStim = 96; % number of stimuli
switch DNNname
    case 'alexnet'
        nLayer = 8;
    case 'alexnet-62'
        nLayer = 8;
        nStim = 62;
end
act = cell(nLayer,1);
indx=1; % for image files
fprintf('Making activation file %s:\n',DNNname);

%cd(fullfile(fileDir,DNNname,sprintf('%s_images%d-%d',DNNname,infile(i,1),infile(i,2))));
cd(fullfile(fileDir,DNNname,'trueActivation'));
d=dir('image*.mat');
for f=1:size(d)
    t=load(d(f).name);
    for n=1:nLayer
        if i==1 && indx==1 % initialise variables
            act{n} = zeros(nStim,length(t.acti{n}(:)));
        end
        act{n}(indx,:) = t.acti{n}(:); % consider the dimensions: images, units, noise
    end
    fprintf('%d.',indx);
    indx=indx+1;
end
% save the new activation file
save(fullfile(fileDir,DNNname,sprintf('%s_trueActivations',DNNname)),'act','-v7.3');
