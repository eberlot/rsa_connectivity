function varargout = alexnet_connect(what,varargin)
baseDir = '/Users/Eva/Documents/Data/rsa_connectivity';
load(fullfile(baseDir,'imageActivations_alexNet_4Eva'),'activations_rand');
act = activations_rand;
clear activations_rand;

switch what
    case 'run_job'
        figOn = 1;
        vararginoptions(varargin,{'figOn'})
        %% 1) estimate first level metrics (G, RDM)
        [G,RDM,U] = alexnet_connect('estimate_firstLevel',figOn);
        %% 2) estimate second level metrics (between RDMs, Gs)
        % 2a) calculate distance based on mean activity (univariate)
        alpha{1} = alexnet_connect('estimate_distance',U,figOn,'univariate');
        % 2b) calculate distance between RDMs (cosine, correlation)
        alpha{2} = alexnet_connect('estimate_distance',RDM,figOn,'RDM'); 
        % partialcorri(RDM(8,:)',RDM([1:7],:)','Rows','complete');
        % 2c) calculate transformation matrices T between Gs
        alpha{3} = alexnet_connect('transformG',G,figOn);
        %% 3) estimate the layer ordering based on distance metrics
        orderDist = alexnet_connect('layer_order',alpha(1:2),'dist');  
       %% 4) estimate the layer ordering based on the transformation metrics
       
       varargout{1}=orderDist;
       % t=getrow(T,T.l1==4&T.l2==6);  % choose layers of interest
       % alexnet_connect('plotTransform',rsa_squareIPMfull(t.T));
        
    case 'estimate_firstLevel'
        %% estimate metrics on first level (i.e. *per* layer)
        % - G matrices per layer
        % - RDM matrices per layer
        % optional: plot RDMs per layer (plots by default)
        % usage: alexnet_connect('estimate_firstLevel',,0);
        fig=varargin{1};
        % 1) estimate G matrices for each layer
        G = alexnet_connect('estimateG'); 
        % 2) estimate RDMs
        RDM = alexnet_connect('estimateRDM',G);
        if fig
            alexnet_connect('plotRDM',RDM);
        end
        % 3) estimate mean patterns - univariate
        U = alexnet_connect('estimateUnivariate');
        varargout{1}=G;
        varargout{2}=RDM;
        varargout{3}=U;
        case 'estimateG'
        % estimates G from input (for each layer)
        nLayer = size(act,1);
        G = cell(nLayer,1);
        for i=1:nLayer
            nVox = size(act{i},2);
            G{i} = act{i}*act{i}'/nVox;
            G{i} = G{i}./trace(G{i});
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
            %plot calculated RDMs 
        RDM = varargin{1};
        nRDM = size(RDM,1);
        figure
        for i=1:nRDM
            subplot(2,4,i);
            imagesc(rsa_squareRDM(RDM(i,:)));
            colormap('hot');
            title(sprintf('RDM-layer%d',i));
        end
        case 'estimateUnivariate'
            % estimate a univariate metric - mean response per condition
        nLayer = size(act,1);
        nStim  = size(act{1},1);
        U = zeros(nLayer,nStim);
        for i=1:nLayer
            U(i,:)=mean(act{i},2)';
        end
        varargout{1}=U;
        
    case 'estimate_distance' 
        %% esimate distance between layers - 2nd level
        RDM = varargin{1};
        fig = varargin{2};
        dataType = varargin{3}; % univariate or multivariate connectivity
        alpha=[];
        aType = {'correlation','cosine'};
        for i=1:2
            D = alexnet_connect('calcDist',RDM,aType{i});
            D.distType = ones(size(D.l1))*i;
            D.distName = repmat(aType(i),size(D.l1));
            alpha = addstruct(alpha,D);
        end
        varargout{1}=alpha;
        if fig
            alexnet_connect('plot_distLayer',alpha,aType,dataType);
        end
        case 'calcDist'
            % calculate correlation / cosine distances
        rdm = varargin{1};
        distType = varargin{2};
        % calculate distance metric from the input
        % input: N x D matrix (N - number of RDMs; D - distance pairs)
        % dist types: 'correlation' or 'cosine'
        % output: structure D, with fields:
        %   - D: pairwise distances of RDMs
        %   - l1: indicator which rdm / layer taken as first
        %   - l2: indicator which rdm ; layer taken as second
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
        [~,D.l1]=find(ind==1);
        i=ismember(ind,-1);
        D.l2 = sum(cumprod(i==0,2),2)+1;
        varargout{1}=D;
        case 'plot_distLayer'
            % plot distances estimated between layers
            alpha=varargin{1};
            aType=varargin{2};
            dataType=varargin{3};
            distType = unique(alpha.distType);
            nDist = numel(distType);
            figure
            for i=distType'
                subplot(nDist,nDist,(i-1)*nDist+1)
                imagesc(rsa_squareRDM(alpha.dist(alpha.distType==i)'));
                colormap('hot');
                title(sprintf('%s acrossLayer: %s distance',dataType,aType{i}));
                [Y,~] =rsa_classicalMDS(rsa_squareRDM(alpha.dist(alpha.distType==i)'));
                subplot(nDist,nDist,(i-1)*2+2)
                scatterplot(Y(:,1),Y(:,2),'label',(1:8));
                hold on;
                drawline(0,'dir','horz');
                drawline(0,'dir','vert');
                title('MDS representation - %s distance');
            end
    case 'transformG'
        %% calculate the transformation matrix between Gs
        G=varargin{1};
        figOn=varargin{2};
        numG = size(G,2);
        TT=[];
        if figOn
            figure
        end
        for i1=1:numG % all pairs (because T is not symmetric)
            for i2=1:numG
                T.l1=i1;
                T.l2=i2;
                [Tr,predG,~,corDist]  = calcTransformG(G{T.l1},G{T.l2}); % can retrieve cosine distance
                T.T             = rsa_vectorizeIPMfull(round(Tr,3));
                T.predG         = rsa_vectorizeIPMfull(predG);
                T.corDist       = corDist;
                T.distType      = ones(size(T.l1));
                TT=addstruct(TT,T);
                if figOn
                    subplot(numG,numG,(i1-1)*numG+i2);
                    imagesc(calcTransformG(G{T.l1},G{T.l2}));
                    title(sprintf('layer: %d-%d',T.l1,T.l2));
                    colorbar;
                end
            end
        end
        TT = alexnet_connect('characterizeT',TT,G,figOn);
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
        case 'characterizeT'
            % characterize different aspects of transformation matrix T
            T=varargin{1};
            G=varargin{2};
            figOn=varargin{3};
            [T.scalar,T.scaleFit,T.scaleDist]   = alexnet_connect('scaleT',T,G);
            T.diagRange                         = alexnet_connect('diagRange',T);
            T.rank                              = alexnet_connect('rank',T);  
            [T.dimFit,T.dimDist,T.dimension]    = alexnet_connect('dimensionT',T,G,figOn);
            varargout{1} = T; % output - added fields characterize T
        case 'scaleT'
            % how different is T from a pure 'scaling' transformation
            T=varargin{1};
            G=varargin{2};
            % initialize outputs
            nPair       = size(T.T,1); % number of compared layers
            scalar      = zeros(nPair,1);
            scaleFit    = zeros(nPair,1);
            scaleDist   = zeros(nPair,1);
            for i=1:nPair
                Tr = rsa_squareIPMfull(T.T(i,:));
                scalar(i) = mean(diag(Tr)); % which scalar
                scaleTr = eye(size(Tr))*scalar(i); % simplified transform - only scalar
                [~,scaleFit(i),scaleDist(i)] = predictGfromTransform(G{T.l1(i)},scaleTr,'G2',G{T.l2(i)});
            end
            varargout{1}=scalar;
            varargout{2}=scaleFit;
            varargout{3}=scaleDist; % correlation distance - how different from a scaled version
        case 'diagRange'
            % calculate the range of diagonal values for each T 
            T=varargin{1};
            nStim           = size(rsa_squareIPMfull(T.T(1,:)),1);
            diagElement     = T.T(:,1:nStim+1:end);
            diagRange       = max(diagElement,[],2)-min(diagElement,[],2);
            varargout{1}    = diagRange;
        case 'rank'
            % calculate the rank of each transformation matrix
            T=varargin{1};
            nPair = size(T.T,1);
            rankT = zeros(nPair,1);
            for i=1:nPair
                rankT(i)=rank(rsa_squareIPMfull(T.T(i,:)));
            end
            varargout{1}=rankT;
        case 'dimensionT'
            % dimensionality of T, and fits of predG with different dim num
            T       = varargin{1};
            G       = varargin{2};
            figOn   = varargin{3};
            % initialize variables
            nPair   = size(T.T,1);
            nStim   = size(rsa_squareIPMfull(T.T(1,:)),1);
            dimFit  = zeros(nPair,nStim);
            dimDist = zeros(nPair,nStim);
            dimension = zeros(nPair,1);
            critFit = 0.9; % criterion applied for fit to assess dimensionality
            for i=1:nPair
                Tr = rsa_squareIPMfull(T.T(i,:));
                % reduce Tr to different dimensions using svd
                [u,s,v]=svd(Tr);
                for j=1:size(u,1)
                    in=1:j;
                    T_reduced = u(:,in)*s(in,in)*v(:,in)';
                    [~,dimFit(i,j),dimDist(i,j)] = predictGfromTransform(G{T.l1(i)},T_reduced,'G2',G{T.l2(i)});
                end
                dimension(i) = find(dimFit(i,:)>critFit,1); % find how many dimensions predict 95 % of G2
            end
            if figOn
                alexnet_connect('plot_dimensionT',dimFit,T.l1,T.l2,critFit);
            end
            varargout{1}=dimFit;
            varargout{2}=dimDist;
            varargout{3}=dimension;
        case 'plot_dimensionT'
            % plot the dimensionality of T
            dimensions  = varargin{1};
            ind1        = varargin{2};
            ind2        = varargin{3};
            cutoff      = varargin{4};
            nDim        = size(dimensions,2);
            
            figure
            subplot(131)
            plot(1:nDim,dimensions(ind1~=ind2,:));
            hold on;
            title('All pairs of layers (both directions)');
            % pick randomly 4 different layers
            subLayer = sample_wor(1:max(ind1),4);
            ind = indicatorMatrix('allpairs',1:4);
            for i=1:size(ind,1)
                subInd1 = subLayer(ind(i,:)==1);
                subInd2 = subLayer(ind(i,:)==-1);
                subplot(132)
                plot(1:nDim,dimensions(ind1==subInd1 & ind2==subInd2,:),'linewidth',3);
                title('L1 < L2');
                hold on;
                subplot(133)
                plot(1:nDim,dimensions(ind1==subInd2 & ind2==subInd1,:),'linewidth',3);
                title('L2 > L1');
                hold on;
            end
            for i=1:3
                subplot(1,3,i)
                drawline(cutoff,'dir','horz');
                xlabel('Dimensions assessed');
                ylabel('corr(trueG - predG)');
            end
            
    case 'layer_order'
            %% estimate the order between layers for a given metric
            alpha=varargin{1};
            var=varargin{2}; % which variable to consider
            nAlpha=size(alpha,2); % number of cells in the cell array
            
            OO=[]; O=[];
            for a=1:nAlpha
                for i=unique(alpha{a}.distType)'
                    A=getrow(alpha{a},alpha{a}.distType==i);
                    [layer1,layer8]=alexnet_connect('determine_borders',A,var);
                    % calculate order from layer1 and layer8
                    order1=alexnet_connect('estimate_order',A,layer1,var);
                    order2=alexnet_connect('estimate_order',A,layer8,var);
                    % determine if matching orders - accuracy
                    O.accu        = sum(order1==fliplr(order2))/length(order1);
                    O.order1      = order1;
                    O.order2      = fliplr(order2);
                    O.distType    = i;
                    O.alphaType   = a;
                    OO=addstruct(OO,O);
                end
            end
            varargout{1}=O;
        case 'determine_borders'
            % determine which layer is first and last
            A=varargin{1};
            var=varargin{2}; % which variable to consider
            [~,j]=max(A.(var));
            varargout{1}=A.l1(j);
            varargout{2}=A.l2(j);
        case 'estimate_order'
            % estimate order from first / last layer
            A   = varargin{1};      % structure with distances
            ind = varargin{2};    % which layer to start from
            var = varargin{3};    % which variable to use as metric
            RDM = rsa_squareRDM((A.(var)(A.l1~=A.l2 & A.l2>A.l1))'); % first reconstruct RDM
            % for now l2 has to be larger than l1 (TO DO - make flexible
            % for T which are not symmetric)
            [~,order]=sort(RDM(ind,:)); % now sort from smaller -> largest dist
            varargout{1}=order;
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

% function corr = detCorr(A,trueOrder)
% % determines correctedness of adjacency matrix given the true order
% nNode = size(A,1); % number of nodes
% correct = zeros(nNode,1);
% for i=1:nNode
%     [~,ind]=sort(A(i,:));
%     pos=find(trueOrder==i);
%     % go to the left
%     leftSide=trueOrder(pos-1:-1:1); %order from closest to furthest dist
%     if length(leftSide)>1
%         lS=zeros(1,length(leftSide));
%         for k=1:length(lS)
%             lS(k)=find(leftSide(k)==ind);
%         end
%         % determine if correct
%         c1=sum(diff(lS)>0)==(length(lS)-1);
%     else
%         c1=1;
%     end
%     % go to the right
%     rightSide=trueOrder(pos+1:end); %order from closest to furthest dist
%     if length(rightSide)>1
%         rS=zeros(1,length(rightSide));
%         for k=1:length(rS)
%             rS(k)=find(rightSide(k)==ind);
%         end
%         % determine if correct
%         c2=sum(diff(rS)>0)==(length(rS)-1);
%     else
%         c2=1;
%     end
%     correct(i)=floor((c1+c2)/2);
% end
% corr=sum(correct)/length(correct);
% end
