function varargout = alexnet_connect(what,varargin)
baseDir = '/Users/Eva/Documents/Data/rsa_connectivity';
load(fullfile(baseDir,'imageActivations_alexNet_4Eva'),'activations_rand');
act = activations_rand;
clear activations_rand; % rename and clear

trueOrder = [5 3 2 4 6 7 1 8]; % for now just a guess
switch what
    case 'run_job'
        % 1) estimate first level metrics (G, RDM)
        [G,RDM] = alexnet_connect('estimate_firstLevel');
        % 2a) calculate distance between RDMs (cosine, correlation)
        alpha = alexnet_connect('estimate_distRDM',RDM,1); % input 2: figure
        % 2b) determine the order of layers (for now just comparing to the 'true' order)
        A=rsa_squareRDM(alpha.dist(alpha.distType==1)'); % adjacency matrix for corrDist
        correct = detCorr(A,trueOrder);
        varargout{1}=correct;
        % 3) calculate transformation matrices T between Gs
        T = alexnet_connect('transformG',G);

        t=getrow(T,T.l1==4&T.l2==6);  % choose layers of interest
        alexnet_connect('plotTransform',rsa_squareIPMfull(t.T));
        
        % 4) quantify Ts how far from 'scaling'; complexity of operation - cosine, svds?
        % 5) use other approach - multivariate dependence measure, but how?
    case 'estimate_firstLevel'
        % estimate metrics on first level (i.e. *per* layer)
        % - G matrices per layer
        % - RDM matrices per layer
        % optional: plot RDMs per layer (plots by default)
        % usage: alexnet_connect('estimate_firstLevel','fig',0);
        fig=1;
        vararginoptions(varargin,{'fig'});
        % 1) estimate G matrices for each layer
        G = alexnet_connect('estimateG'); 
        % 2) estimate RDMs
        RDM = alexnet_connect('estimateRDM',G);
        if fig
            alexnet_connect('plotRDM',RDM);
        end
        varargout{1}=G;
        varargout{2}=RDM;
        case 'estimateG'
        % estimates G from input (for each layer)
        nLayer = size(act,1);
        G = cell(nLayer,1);
        for i=1:nLayer
            nVox = size(act{i},2);
            G{i} = act{i}*act{i}'/nVox;
        end
        varargout{1}=vout(G);
        case 'estimateRDM'
        % estimates RDM from input G matrices (for each layer)
        G=varargin{1}; % input: G matrices (per layer)
        nLayer = size(G,2);
        nCond = size(G{1},1);
        % contrast matrix for G->distances
        C = indicatorMatrix('allpairs',1:nCond);
        RDM = zeros(nLayer,size(C,1));
        for i=1:nLayer
            RDM(i,:) = diag(C*G{i}*C')';  
        end
        varargout{1}=RDM;
        case 'plotRDM'
        RDM = varargin{1};
        nRDM = size(RDM,1);
        figure
        for i=1:nRDM
            subplot(2,4,i);
            imagesc(rsa_squareRDM(RDM(i,:)));
            colormap('hot');
            title(sprintf('RDM-layer%d',i));
        end
    case 'estimate_distRDM'
        RDM = varargin{1};
        fig = varargin{2};
        alpha=[];
        aType = {'correlation','cosine'};
        for i=1:2
            D = alexnet_connect('calcDist',RDM,aType{i});
            D.distType = ones(size(D.ind1))*i;
            D.distName = repmat(aType(i),size(D.ind1));
            alpha = addstruct(alpha,D);
        end
        varargout{1}=alpha;
        if fig
            figure
            for i=1:2
                subplot(1,2,i)
                imagesc(rsa_squareRDM(alpha.dist(alpha.distType==i)'));
                colormap('hot');
                title(sprintf('acrossLayer distance:%s',aType{i}));
            end
        end
        case 'calcDist'
        rdm = varargin{1};
        distType = varargin{2};
        % calculate distance metric from the input
        % input: N x D matrix (N - number of RDMs; D - distance pairs)
        % dist types: 'correlation' or 'cosine'
        % output: structure D, with fields:
        %   - D: pairwise distances of RDMs
        %   - ind1: indicator which rdm taken as first
        %   - ind2: indicator which rdm taken as second
        switch (distType)
            case 'correlation'
                % additional step for correlation - first remove the mean
                rdm  = bsxfun(@minus,rdm,mean(rdm,2));
        end
        nRDM = size(rdm,1);
        rdm  = normalizeX(rdm);
        tmpR  = rdm*rdm'; % correlation across RDMs
        D.dist = 1-rsa_vectorizeRDM(tmpR)'; % distances
        ind=indicatorMatrix('allpairs',(1:nRDM));
        % determine each element of the pair
        [~,D.ind1]=find(ind==1);
        i=ismember(ind,-1);
        D.ind2 = sum(cumprod(i==0,2),2)+1;
        varargout{1}=D;
    case 'transformG'
        G=varargin{1};
        numG = size(G,2);
        pairG = indicatorMatrix('allpairs',1:numG);
        TT=[];
        figure;
        for i=1:size(pairG,1)
            T.l1=find(pairG(i,:)==1);
            T.l2=find(pairG(i,:)==-1);
            T.T = rsa_vectorizeIPMfull(round(calcTransformG(G{T.l1},G{T.l2}),3));
            subplot(4,7,i)
            imagesc(calcTransformG(G{T.l1},G{T.l2}));
            title(sprintf('layer: %d-%d',T.l1,T.l2));
            colorbar;
            TT=addstruct(TT,T);
        end
        varargout{1}=TT;
        case 'plotTransform'
        % examine how T looks like
        T=varargin{1};
        figure
        subplot(131)
        imagesc(T);
        title('Transformation T');
        [u,s,~]=svd(T);
        subplot(132)
        imagesc(u);
        title('SVD component (u)');
        subplot(133)
        imagesc(s);
        title('SVD component (s)');
end
end

% local functions
function vout = vout(IN)
% function vout = vout(IN)
% provides cell outputs
nOUT = size(IN,1);
vout = cell(1,nOUT);
for i = 1:nOUT
    vout{i} = IN{i};
end
end

function corr = detCorr(A,trueOrder)
% determines correctedness of adjacency matrix given the true order
nNode = size(A,1); % number of nodes
correct = zeros(nNode,1);
for i=1:nNode
    [~,ind]=sort(A(i,:));
    pos=find(trueOrder==i);
    % go to the left
    leftSide=trueOrder(pos-1:-1:1); %order from closest to furthest dist
    if length(leftSide)>1
        lS=zeros(1,length(leftSide));
        for k=1:length(lS)
            lS(k)=find(leftSide(k)==ind);
        end
        % determine if correct
        c1=sum(diff(lS)>0)==(length(lS)-1);
    else
        c1=1;
    end
    % go to the right
    rightSide=trueOrder(pos+1:end); %order from closest to furthest dist
    if length(rightSide)>1
        rS=zeros(1,length(rightSide));
        for k=1:length(rS)
            rS(k)=find(rightSide(k)==ind);
        end
        % determine if correct
        c2=sum(diff(rS)>0)==(length(rS)-1);
    else
        c2=1;
    end
    correct(i)=floor((c1+c2)/2);
end
corr=sum(correct)/length(correct);
end
