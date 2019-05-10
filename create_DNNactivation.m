function create_DNNactivation(DNNname)
%function create_DNNactivation(DNNname)
%creates activation files for DNN
% INPUT:    DNNname (vgg or resnet)

fileDir = '/Users/Eva/Documents/Data/rsa_connectivity';
nStim = 96; % number of stimuli
switch DNNname
    case 'vgg16'
        infile = [1 48; 49 96];
        nLayer = 16;
    case 'resnet50'
        infile = [1 32; 33 64; 65 96];
        nLayer = 50;
    case 'alexnet_imagenet'
        infile = [1 92];
        nLayer = 8;
        nStim = 72;
     case 'alexnet'
        infile = [1 96];
        nLayer = 8;
end
act = cell(nLayer,1);
indx=1; % for image files
fprintf('Making activation file %s:\n',DNNname);
for i=1:size(infile,1)
    cd(fullfile(fileDir,DNNname,sprintf('%s_images%d-%d',DNNname,infile(i,1),infile(i,2))));
    d=dir('image*.mat');
    for f=1:size(d)
        t=load(d(f).name);
        for n=1:nLayer
            if i==1 && indx==1 % initialise variables
                act{n} = zeros(nStim,length(t.acti{n}(:)));
            end
            act{n}(indx,:) = t.acti{n}(:);
        end
        fprintf('%d.',indx);
        indx=indx+1;
    end
end
% save the new activation file
save(fullfile(fileDir,DNNname,sprintf('%s_activations',DNNname)),'act','-v7.3');
