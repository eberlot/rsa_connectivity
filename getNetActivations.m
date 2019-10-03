function getNetActivations(netStr,imagePath,resultsPath,removePixel)

% function getNetActivations(netStr,imagePath,resultsPath,0)

% INPUT
%           - netStr (options: 'alexnet','resnet50')
%           - imagePath (path where original images are contained)
%           - resultsPath (path to save the results)
%           - removePixel (how many pixels to remove in proportion (0-1); 0-no noise)


%% preparation
if ~exist('netStr','var'), netStr = 'alexnet'; end
if strcmp(netStr,'alexnet'), layerIs = [2 6 10 12 14 17 20 23]; 
elseif strcmp(netStr,'resnet50'), layerIs = [2 6 9 12 13 18 21 24 28 31 34 38 41 44 45 50 53 56 60 63 66 70 73 76 80 83 86 87 92 95 98 102 105 108 112 115 118 122 125 128 132 135 138 142 145 148 149 154 157 160 164 167 170 175]; end
 
if ~exist(resultsPath,'dir'); mkdir(resultsPath); end
% remove this
%% control variables
monitor = 0;
distMeasure = 'correlation';

%% load network
if strcmp(netStr,'alexnet'), net = alexnet;
elseif strcmp(netStr,'resnet50'), net = resnet50; end
net.Layers; % print layer info
inputSize = net.Layers(1).InputSize;

%% load & prepare images
% load images
files = get_files(imagePath,'*.BMP');
nImages = size(files,1);
images = nan(inputSize(1),inputSize(2),inputSize(3),nImages); % assume images are RGB
images = single(images);
for imageI = 1:nImages
    im = imread(files(imageI,:));
    if monitor, figure(10); imshow(im); title('original'); end
    % resize image if needed
    if size(im,1) ~= inputSize(1) || size(im,2) ~= inputSize(2)
        im_resized = imresize(im,[inputSize(1) inputSize(2)]);
        if monitor, figure(11); imshow(im_resized); title('resized'); end
    else
        im_resized = im;
    end
    % here eliminate pixels if chosen
    if removePixel>0
        numPix = round(size(im_resized,1).*removePixel);
        rPix = randi(size(im_resized,1),numPix,1);
        im_resized(rPix,rPix,:) = 0;
    end
    % zerocenter image
    im_resized_cntrd = activations(net,im_resized,net.Layers(1).Name); % assuming first layer performs 'zerocenter' normalisation
    if monitor, figure(12); imshow(uint8(im_resized_cntrd)); title('after zerocenter normalisation'); end
    % store image
    images(:,:,:,imageI) = single(im_resized_cntrd);
end % imageI

% save
save(fullfile(resultsPath,'All_images'),'images');
%% compute activations
nLayers = numel(layerIs);
nUnits = nan(nLayers,1);
fileID = fopen(fullfile(resultsPath,'top5.txt'),'w');
for imageI = 1:nImages
    im = images(:,:,:,imageI);    
    acti = cell(nLayers,1);
    for layerIsI = 1:nLayers
        layerI = layerIs(layerIsI);
        layerName = net.Layers(layerI).Name;
        acti{layerIsI} = activations(net,im,layerName);
        nUnits(layerIsI) = numel(acti{layerIsI});
    end % layerI
    [label,scores] = classify(net,im);
    [~,idx] = sort(scores,'descend');
    idx = idx(1:5);
    top5 = net.Layers(end).ClassNames(idx);
    [pth,fle,ext] = fileparts(files(imageI,:));
    save(fullfile(resultsPath,fle),'im','acti','scores','top5');
    % print top5 to text file
    formatSpec1 = '%s\n'; formatSpec2 = '%s\n\n';
    fprintf(fileID,formatSpec1,[fle,ext]); 
    for labelI = 1:numel(idx) 
        if labelI < numel(idx), fprintf(fileID,formatSpec1,top5{labelI}); 
        else fprintf(fileID,formatSpec2,top5{labelI}); end 
    end
end % imageI
save(fullfile(resultsPath,'vars'),'nImages','nLayers','nUnits','files');
fclose(fileID);


% %% compute RDMs % - commented out because it takes time
% for layerIsI = 1:nLayers    
%     layerI = layerIs(layerIsI);
%     acti__images_units = single(nan(nImages,nUnits(layerIsI)));
%     for imageI = 1:nImages
%         [pth,fle,ext] = fileparts(files(imageI,:));
%         load(fullfile(resultsPath,fle),'acti');
%         acti__images_units(imageI,:) = acti{layerIsI}(:);
%     end % imageI
%     cnnRDM(layerIsI).RDM = pdist(acti__images_units,distMeasure);
%     cnnRDM(layerIsI).name = net.Layers(layerI).Name;    
% end % layerI
% save(fullfile(resultsPath,['RDMs_',distMeasure]),'cnnRDM');


