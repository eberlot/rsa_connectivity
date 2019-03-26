function varargout = alexnet_connect(what,varargin)
% function varargout = alexnet_connect(what,varargin)
% 1) Calculates metrics of 'connectivity' on alexnet activation units
% 2) Determines the order of layers based on the calculated metrics
% 3) Plots the estimated order (and intermediate steps) - optional
% usage to run all functionality: alexnet_connect('run_all','figOn',1);
%
% INPUT:
%       - case: which case to run; if all: alexnet_connect('run_all');
%       - orderUse: random or correct (which order to use)
% VARARGIN:
%       - baseDir: where the alexNet mat file with activations is stored
%       - figOn: whether intermediate results are plotted (0/1; default:1)
%
% OUTPUTS:
%       - alexnet_RDM:      matrix with vectorized RDM for each layer
%                           (size: numLayer x numDist)
%       - alexnet_G:        cell array with G for each layer stored as a matrix
%       - alexnet_alpha:    structure array with three structures:
%                           1) univariate metrics of distances
%                           2) multivariate metrics of distances (RDM)
%                           3) transformation metrics of G matrices
%       - order_undirected: structure with the estimated layer order for
%                           undirected metrics (alpha uni and multivariate)
%       - order_undirected: structure with the estimated layer order for
%                           directed metrics (transformation between Gs)
%       *note*: all outputs are saved in the baseDir
%
% usage: alexnet_connect('run_all','figOn',1);
%
baseDir = '/Volumes/MotorControl/data/rsa_connectivity/alexnet';
load(fullfile(baseDir,'imageActivations_alexNet_4Eva'),'activations_rand','activations_correct');
load(fullfile(baseDir,'imageAct_subsets'),'act_subsets'); 
randOrder = [8 1 7 6 4 2 3 5]; % how the order was first determined - using activations_rand - double blind procedure
correctOrder = 1:8; % correct order
numLayer = 8;
actUse = 'subsets'; % here change random or correct
mColor={[84 13 100]/255,[238 66 102]/255,[14 173 105]/255,[59 206 172]/255,[255 210 63]/255,[78 164 220]/255,[176 0 35]/255,[170 170 170]/255};
aType = {'correlation','cosine'};

if strcmp(actUse,'correct') % choose the ordering
    order = correctOrder;
    act = activations_correct;
elseif strcmp(actUse,'random')
    order = randOrder;
    act = activations_rand;
elseif strcmp(actUse,'subsets')
    order = correctOrder;
    act = act_subsets;
else
    error('wrong option!\n');
end
clear activations_rand; clear activations_correct;

switch what
    case 'run_all'
        figOn = 1;
        nameFile = sprintf('alexnet_%s',actUse);
        %% run the whole script 
        vararginoptions(varargin,{'figOn','nameFile'});
        alexnet_connect('run_calcConnect','figOn',figOn,'nameFile',nameFile);
        alexnet_connect('run_deriveOrder');
        if figOn
            alexnet_connect('plot_estimatedOrder');
        end
    case 'run_calcConnect'
        % gets the first and second level metrics
        % first level: G, RDM
        % second level: univariate / RDM connect, transformation of Gs
        % saves the outputs
        figOn = 1;
        nameFile = sprintf('alexnet_%s',actUse);
        vararginoptions(varargin,{'figOn','nameFile'});
        %% 1) estimate first level metrics (G, RDM)
        [G,RDM,U] = alexnet_connect('estimate_firstLevel',figOn);
        %% 2) estimate second level metrics (between RDMs, Gs)
        % 2a) calculate distance based on mean activity (univariate)
        alpha{1} = alexnet_connect('estimate_distance',U,figOn,'univariate');
        % 2b) calculate distance between RDMs (cosine, correlation)
        alpha{2} = alexnet_connect('estimate_distance',RDM,figOn,'RDM'); 
        % 2c) calculate transformation matrices T between Gs
        alpha{3} = alexnet_connect('transformG',G,figOn);
        varargout{1}=alpha;
        %% save the variables
        save(fullfile(baseDir,sprintf('%s_G',nameFile)),'G');
        save(fullfile(baseDir,sprintf('%s_univariate',nameFile)),'U');
        save(fullfile(baseDir,sprintf('%s_RDM',nameFile)),'RDM');
        save(fullfile(baseDir,sprintf('%s_alpha',nameFile)),'alpha');
    case 'run_deriveOrder'
        % calculate the order of layers based on metrics
        a=load(fullfile(baseDir,'alexnet_alpha'),'alpha'); 
        alpha=a.alpha;
        %% 1) estimate the layer ordering based on distance metrics (undirected)
        O_und = alexnet_connect('layer_order_undirected',alpha(1:2),'dist');
        %% 2) determine which are the most likely borders
        [b1,b2]=alexnet_connect('mostLikely_borders',O_und);
        %% 3) calculate the directed flow
        %metrics={'corDist','scaleDist','diagDist','diagRange','dimension'};
        metrics={'corDist','scaleFit','diagFit','diagRange','eigStd','dimension'};
        O_dir = alexnet_connect('layer_order_directed',alpha{3},metrics,b1,b2);
        %% 4) calculate the order with the correct start
        O_uStart = alexnet_connect('layer_order_undir_correctStart',alpha(1:2),'dist');
        O_dStart = alexnet_connect('layer_order_dir_correctStart',alpha{3},metrics);
        %% 5) save the estimated order of layers (undirected and directed)
        save(fullfile(baseDir,'order_undirected'),'-struct','O_und');
        save(fullfile(baseDir,'order_directed'),'-struct','O_dir');
        save(fullfile(baseDir,'order_undir_corStart'),'-struct','O_uStart');
        save(fullfile(baseDir,'order_dir_corStart'),'-struct','O_dStart');
        varargout{1}={O_und O_dir O_uStart O_dStart};
        case 'estimate_accuracy'
            % estimate the accuracy of the determined order
            orderName = {'undir_corStart','dir_corStart'};
            for i = 1:length(orderName)
                T = load(fullfile(sprintf('order_%s',orderName{i})));
                order = T.order;
                order(T.typeOrder==3,:) = fliplr(T.order(T.typeOrder==3,:));
                T.corrElem    = zeros(size(T.typeOrder));
                T.nSteps      = zeros(size(T.typeOrder));
                for j=1:size(order,1)
                    if T.typeOrder(j)==3 % flip and start from the end
                        tmpOrder = fliplr(order);
                    else
                        tmpOrder = order;
                    end
                    T.corrElem(j) = (sum(order(j,2:end)==order(2:end)))/(size(order,2)-1);
                    % determine how many steps away
                    nstep=[];
                    for k=2:length(order)
                        pos=find(order(j,:)==tmpOrder(k));
                        nstep = [nstep; pos-k];
                    end
                    T.nSteps(j) = mean(abs(nstep));
                end
                save(fullfile(baseDir,sprintf('order_%s',orderName{i})),'-struct','T');
                % K=tapply(T,{'distName','alphaType'},{'corrElem','mean'});
            end
        case 'plot_estimatedOrder'
        %% plots the estimated order of layers
        U = load(fullfile(baseDir,'order_undirected'));
        alexnet_connect('plot_order_undirected',U);
        D = load(fullfile(baseDir,'order_directed'));
        alexnet_connect('plot_order_directed',D);
        US = load(fullfile(baseDir,'order_undir_corStart'));
        alexnet_connect('plot_order_undir_correctStart',US);
        DS = load(fullfile(baseDir,'order_dir_corStart'));
        alexnet_connect('plot_order_dir_correctStart',DS);
    case 'run_cluster'
        % derive relationship between clusters (each layer split into 4 regions)
        a = load(fullfile(baseDir,'alexnet_subsets_alpha'),'alpha'); 
        C = a.alpha; % cluster alpha values
        %% 1) estimate clusters based on distance metrics - undirected
        C_und = alexnet_connect('cluster_undirected',C(1:2),'dist');
        %% 2) directed metrics
        metrics={'corDist','scaleFit','diagFit','diagRange','eigStd','dimension'};
        C_dir = alexnet_connect('cluster_directed',C{3},metrics);
        save(fullfile(baseDir,'cluster_undirected'),'-struct','C_und');
        save(fullfile(baseDir,'cluster_directed'),'-struct','C_dir');
        varargout{1}={C_und C_dir};
        case 'plot_estimatedClusters'
            %% here plot the estimated clusters (into layers, their relationship)
            U = load(fullfile(baseDir,'cluster_undirected'));
            alexnet_connect('plot_cluster_undirected',U);
            D = load(fullfile(baseDir,'cluster_directed'));
            alexnet_connect('plot_cluster_directed',D);
            
    case 'plot_metricRelations'
        T=load(fullfile(baseDir,'alexnet_alpha'));
        D1=T.alpha{1};
        D2=T.alpha{2};
        D3=T.alpha{3};
        alexnet_connect('plot_metrics_undir',D1,D2);
        alexnet_connect('plot_metrics_dir',D3); % add the new ones - diag-offdiag, range vs. scalar
        keyboard;
        %%
    
    case 'estimate_firstLevel'
        %% estimate metrics on first level (i.e. *per* layer)
        % - G matrices per layer
        % - RDM matrices per layer
        % optional: plot RDMs per layer (plots by default)
        % calculate everything per layer or per cluster
        % usage: alexnet_connect('estimate_firstLevel',0,'clust');
        fig=varargin{1};
        % 1) estimate G matrices for each layer
        G = alexnet_connect('estimateG',act); 
        % 2) estimate RDMs
        RDM = alexnet_connect('estimateRDM',G);
        if fig
            alexnet_connect('plotRDM',RDM);
        end
        % 3) estimate mean patterns - univariate
        U = alexnet_connect('estimateUnivariate',act);
        varargout{1}=G;
        varargout{2}=RDM;
        varargout{3}=U;
        fprintf('Done estimating first level metrics: uni, RDM, G.\n');
        case 'estimateG'  
            % estimates G from input (for each layer)
            act = varargin{1};
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
                subplot(4,nRDM/4,i);
                imagesc(rsa_squareRDM(RDM(i,:)));
                colormap('hot');
                title(sprintf('RDM-layer%d',i));
            end
        case 'estimateUnivariate'
            % estimate a univariate metric - mean response per condition
            act    = varargin{1};
            nLayer = size(act,1);
            nStim  = size(act{1},1);
            U = zeros(nLayer,nStim);
            for i=1:nLayer
                t=bsxfun(@minus,act{i},mean(act{i},1));  % first here remove the mean activation
                U(i,:)=mean(t,2)'; 
                %U(i,:)=mean(act{i},2)';
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
        fprintf('Done estimating undirected distance metrics: %s.\n',dataType);
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
                title(sprintf('MDS representation - %s distance',aType{i}));
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
        fprintf('Done estimating transformation between G matrices.\n');
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
            [T.diagFit,T.diagDist]              = alexnet_connect('diagT',T,G); 
            T.diagRange                         = alexnet_connect('diagRange',T);
            T.rank                              = alexnet_connect('rank',T);
            [T.eig,T.eigStd]                    = alexnet_connect('eigenvalues',T);
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
                Tr          = rsa_squareIPMfull(T.T(i,:));
                scalar(i)   = mean(diag(Tr)); % which scalar
                scaleTr     = eye(size(Tr))*scalar(i); % simplified transform - only scalar
                [~,scaleFit(i),scaleDist(i)] = predictGfromTransform(G{T.l1(i)},scaleTr,'G2',G{T.l2(i)});
            end
            varargout{1}=scalar;
            varargout{2}=scaleFit;
            varargout{3}=scaleDist; % correlation distance - how different from a scaled version
        case 'diagT'
            % how different is T from a pure diagonal T transformation
            T=varargin{1};
            G=varargin{2};
            % initialize outputs
            nPair       = size(T.T,1); % number of compared layers
            diagFit     = zeros(nPair,1);
            diagDist    = zeros(nPair,1);
            for i=1:nPair
                Tr          = rsa_squareIPMfull(T.T(i,:));
                diagTr      = Tr.*eye(size(Tr));
                [~,diagFit(i),diagDist(i)] = predictGfromTransform(G{T.l1(i)},diagTr,'G2',G{T.l2(i)});
            end
            varargout{1}=diagFit;
            varargout{2}=diagDist; % correlation distance - how different from a scaled version
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
        case 'eigenvalues'
            % determine the range of eigenvalues of transformation matrix T
            % returns the eigenvalues and their variance
            T=varargin{1};
            nPair   = size(T.T,1);
            eigT    = zeros(nPair,size(rsa_squareIPMfull(T.T(1,:)),1));
            eigStd  = zeros(nPair,1);
            for i=1:nPair
                Tr = rsa_squareIPMfull(T.T(i,:));
                eigT(i,:)   = eig(Tr)';
                eigStd(i)   = std(eigT(i,:));
            end
            varargout{1}=eigT;
            varargout{2}=eigStd;
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
            
    case 'layer_order_undirected'
        %% estimate the order between layers for a given metric (undirected)
        alpha   = varargin{1};
        var     = varargin{2}; % which variable to consider
        nAlpha  = size(alpha,2); % number of cells in the cell array 
        OO=[]; 
        for a=1:nAlpha
            for i=unique(alpha{a}.distType)'
                A=getrow(alpha{a},alpha{a}.distType==i);
                [layer1,layer8]=alexnet_connect('determine_borders',A,var);
                % calculate order from layer1 and layer8
                O.order1 = alexnet_connect('estimate_order_undir',A,layer1,var);
                O.order2 = fliplr(alexnet_connect('estimate_order_undir',A,layer8,var));
                % determine if matching orders - accuracy
                O.accu        = sum(O.order1==O.order2)/length(O.order1);
                O.distType    = i;
                O.distName    = {var};
                O.alphaType   = a;
                % here determine the pairwise (neighbour) distances between layers
                O.nDist1 = alexnet_connect('neighbour_distances',A,O.order1,var,'undirected');
                O.nDist2 = alexnet_connect('neighbour_distances',A,O.order2,var,'undirected');
                OO=addstruct(OO,O);
            end
        end
        fprintf('Done estimating order of layers: undirected.\n');
        varargout{1}=OO;
    case 'layer_order_directed'
        %% estimate the order between layers for a given metric (directed)
        alpha   = varargin{1};
        var     = varargin{2}; % which variable to consider
        b1      = varargin{3}; % most likely borders
        b2      = varargin{4};        
        % estimate in two ways: 1) most likely borders, 2) metric-specified
        O_dir = [];
        for m=1:length(var)
            for b=1:2 % border type
                if b==2 % specify by metric
                    [B1,B2] = alexnet_connect('determine_borders',alpha,var{m});
                else
                    B1=b1;
                    B2=b2;
                end
                d1 = getrow(alpha,alpha.l1==B1 & alpha.l2==B2);
                d2 = getrow(alpha,alpha.l1==B2 & alpha.l2==B1);
                if d1.(var{m}) > d2.(var{m}) % should be more difficult to get from first to last, then the opposite
                    O.order = alexnet_connect('estimate_order_dir',alpha,B1,var{m});
                else
                    O.order = alexnet_connect('estimate_order_dir',alpha,B2,var{m});
                end
                O.distType    = m;
                O.distName    = {var{m}};
                O.alphaType   = 3;
                O.borderType  = b;
                % here determine the pairwise (neighbour) distances between layers
                O.nDist = alexnet_connect('neighbour_distances',alpha,O.order,var{m},'directed');
                O_dir = addstruct(O_dir,O);
            end
        end
        fprintf('Done estimating order of layers: directed.\n');
        varargout{1}=O_dir;
    case 'layer_order_undir_correctStart'
        % determine the order of metrics given the correct start
        alpha       = varargin{1};
        var         = varargin{2}; % which variable to consider
        nAlpha      = size(alpha,2); % number of cells in the cell array 
        OO=[];
        for type=1:3 % type 1: from node 1, type 2: neighbours, type 3: from end
            for a=1:nAlpha
                for i=unique(alpha{a}.distType)'
                    A=getrow(alpha{a},alpha{a}.distType==i);
                    if type==1
                        O.order = alexnet_connect('estimate_order_undir',A,order(1),var);
                    elseif type==2
                        O.order = alexnet_connect('estimate_order_neighbours_undir',A,order(1),var);
                    else
                        O.order = alexnet_connect('estimate_order_neighbours_undir',A,order(8),var);
                    end
                    % determine if matching orders to the correct one - accuracy
                    O.accu        = sum(O.order==order)/length(O.order);
                    O.distType    = i;
                    O.distName    = {A.distName{1}};
                    O.alphaType   = a;
                    O.typeOrder   = type;
                    % here determine the pairwise (neighbour) distances between layers
                    O.nDist = alexnet_connect('neighbour_distances',A,O.order,var,'undirected');
                    OO=addstruct(OO,O);
                end
            end
        end
        varargout{1}=OO;
    case 'layer_order_dir_correctStart'
        % determine the order of metrics given the correct start
        A   = varargin{1};
        var = varargin{2}; % which variable(s) to consider
        OO=[];
        for type=1:3 % type 1: from node 1, type 2: neighbours (from start), type 3: from end
            for m=1:length(var)
                if type==1
                    t = getrow(A,A.l1==order(1) & A.l1~=A.l2);
                    [~,order]       = sort(t.(var{m}));
                    O.order         = [order(1) order'];
                elseif type==2
                    O.order = alexnet_connect('estimate_order_dir',A,order(1),var{m});
                else
                    O.order = alexnet_connect('estimate_order_dir',A,order(8),var{m});
                end
                O.distType      = m;
                O.distName      = {var{m}};
                O.alphaType     = 3;
                O.typeOrder     = type;
                O.nDist         = alexnet_connect('neighbour_distances',A,O.order,var{m},'directed');
                OO = addstruct(OO,O);
            end
        end
        varargout{1}=OO;
        case 'determine_borders'
            % determine which layer is first and last
            A=varargin{1};
            var=varargin{2}; % which variable to consider
            [~,j]=max(A.(var));
            varargout{1}=A.l1(j);
            varargout{2}=A.l2(j);
        case 'estimate_order_undir'
            % estimate order from first / last layer - undirected
            A   = varargin{1};    % structure with distances
            ind = varargin{2};    % which layer to start from
            var = varargin{3};    % which variable to use as metric
            RDM = rsa_squareRDM(A.(var)'); % first reconstruct RDM
            [~,order]=sort(RDM(ind,:)); % now sort from smaller -> largest dist
            varargout{1}=order;
        case 'estimate_order_dir'
            % estimate order from first / last layer - directed
            A   = varargin{1};    % structure with distances
            ind = varargin{2};    % which layer to start from
            var = varargin{3};    % which variable to use as metric
            order = zeros(1,max(A.l1));
            order(1)=ind;
            for i = 2:length(order)
                t = getrow(A,A.l1==order(i-1));
                [~,ord]=sort(t.(var));
                inter = intersect(order,ord);
                ord = ord(~ismember(ord,inter),:); % remove any previously chosen elements
                order(i) = ord(1);
            end
            varargout{1}=order;
        case 'estimate_order_neighbours_undir'
            % estimate order from first / last layer - directed
            A   = varargin{1};    % structure with distances
            ind = varargin{2};    % which layer to start from
            var = varargin{3};    % which variable to use as metric
            RDM = rsa_squareRDM(A.(var)'); % matrix format
            order = zeros(1,size(RDM,1));
            order(1)=ind;

            for i = 2:length(order)
                [~,ord]=sort(RDM(order(i-1),:));
                inter = intersect(order,ord);
                ord = ord(~ismember(ord,inter)); % remove any previously chosen elements
                order(i) = ord(1);
            end
            varargout{1}=order;
        case 'mostLikely_borders'
            O = varargin{1}; % order structure (across different metrics)
            border=[O.order1(:,1);O.order2(:,1);O.order1(:,end);O.order2(:,end)];
            [n,bin] = hist(border,unique(border));
            [~,idx] = sort(-n); % swap the order
            % corresponding 2 most common bordervalues
            b1=bin(idx(1));
            b2=bin(idx(2));
            varargout{1}=b1;
            varargout{2}=b2;
        case 'neighbour_distances'
            % determine the neighbouring distances for the estimated
            % ordering of layers
            alpha = varargin{1};
            order = varargin{2};
            var   = varargin{3};
            graphType = varargin{4}; % directed or undirected
            nDist = zeros(1,length(order)-1); % neighbour distance pre-alloc
            for i=1:length(nDist)
                switch graphType
                    case 'directed'
                        nDist(i)=alpha.(var)(alpha.l1==order(i) & alpha.l2==order(i+1));
                    case 'undirected'
                        nDist(i)=alpha.(var)(ismember(alpha.l1,[order(i),order(i+1)]) & ismember(alpha.l2,[order(i),order(i+1)]));
                end
            end
            varargout{1}=nDist;
 
    case 'plot_order_undirected'
        % plot the order for undirected graph
        T=varargin{1}; % data structure
        distType={'univariate','multivariate'};
        alphaType={'correlation','cosine'};
        for d=1:length(distType)
            figure
            indx=1;
            for m=1:length(alphaType)
                t   = getrow(T,T.distType==d & T.alphaType==m);
                subplot(2,2,indx)
                pos = [0 cumsum(t.nDist1)];
                scatterplot(pos',zeros(size(pos))','label',t.order1,'markersize',8);
                drawline(0,'dir','horz');
                title(sprintf('%s-%s direction 1',distType{d},alphaType{m}));
                subplot(2,2,indx+1)
                pos = [0 cumsum(t.nDist2)];
                scatterplot(pos',zeros(size(pos))','label',t.order2,'markersize',8);
                drawline(0,'dir','horz');
                title(sprintf('%s-%s direction 2',distType{d},alphaType{m}));
                indx = indx+2;
            end
        end
    case 'plot_order_directed'
        % plot the order for directed graph
        TT=varargin{1}; % data structure
        distName=unique(TT.distName);
        borderType={'external','self-determined'};
        for b=1:2
            T=getrow(TT,TT.borderType==b);
            figure
            for d=1:length(distName)
                t   = getrow(T,strcmp(T.distName,distName{d}));
                subplot(length(distName),1,d);
                pos = [0 cumsum(t.nDist)];
                scatterplot(pos',zeros(size(pos))','label',t.order,'markersize',8);
                drawline(0,'dir','horz');
                title(sprintf('%s - %s borders',distName{d},borderType{b}));
            end
        end
    case 'plot_order_undir_correctStart'
        T=varargin{1}; % data structure
        dataType={'univariate','multivariate'};
        distName=unique(T.distName);
        for to = 1:length(unique(T.typeOrder))
            for dt = 1:length(dataType)
                figure
                for d=1:length(distName)
                    t   = getrow(T,strcmp(T.distName,distName{d}) & T.alphaType==dt & T.typeOrder==to);
                    subplot(length(distName),1,d);
                    pos = [0 cumsum(t.nDist)];
                    scatterplot(pos',zeros(size(pos))','label',t.order,'markersize',8);
                    drawline(0,'dir','horz');
                    title(sprintf('%s %s - type order-%d',dataType{dt},distName{d},to));
                end
            end
        end
    case 'plot_order_dir_correctStart'
          T=varargin{1}; % data structure
          distName=unique(T.distName);
          for to=1:length(unique(T.typeOrder))
              figure
              for d=1:length(distName)
                  t   = getrow(T,strcmp(T.distName,distName{d}) & T.typeOrder==to);
                  subplot(length(distName),1,d);
                  pos = [0 cumsum(t.nDist)];
                  scatterplot(pos',zeros(size(pos))','label',t.order,'markersize',8);
                  drawline(0,'dir','horz');
                  title(sprintf('%s - type order-%d',distName{d},to));
              end
          end
    case 'plot_metrics_undir'
        D1=varargin{1}; % univariate
        D2=varargin{2}; % multivariate
        figure
        subplot(221)
        plt.scatter(D1.dist(D1.distType==1),D1.dist(D1.distType==2));
        xlabel(unique(D1.distName(D1.distType==1)));ylabel(unique(D1.distName(D1.distType==2)));
        title('univariate only');
        subplot(222)
        plt.scatter(D2.dist(D2.distType==1),D2.dist(D2.distType==2));
        xlabel(unique(D2.distName(D2.distType==1)));ylabel(unique(D2.distName(D2.distType==2)));
        title('multivariate only');
        subplot(223)
        plt.scatter(D1.dist(D1.distType==1),D2.dist(D2.distType==1));
        xlabel('correlation uni');ylabel('correlation multi');
        title('across');
        subplot(224)
        plt.scatter(D1.dist(D1.distType==2),D2.dist(D2.distType==2));
        xlabel('cosine uni');ylabel('cosine multi');
        title('across');
    case 'plot_metrics_dir'
        D=varargin{1}; % directed metrics
        keyboard;
        metrics={'corDist','scaleDist','diagRange','eigStd','dimension'};
        ind=indicatorMatrix('allpairs',1:length(metrics));
        figure
        for i=1:size(ind,1)
            subplot(2,3,i)
            plt.scatter(D.(metrics{ind(i,:)==1}),D.(metrics{ind(i,:)==-1}),'subset',D.l1~=D.l2);
            xlabel(metrics{ind(i,:)==1});ylabel(metrics{ind(i,:)==-1});
        end
        
    case 'plot_dimensions'
        a=load(fullfile(baseDir,'alexnet_alpha'));
        G=load('alexnet_G');
        G=G.G;
        t=a.alpha{3};
        for i=1:length(order)-1 % inspect the neighbours
            T_for   = rsa_squareIPMfull(t.T(t.l1==order(i)&t.l2==order(i+1),:)); % forward transformation
            T_back  = rsa_squareIPMfull(t.T(t.l1==order(i+1)&t.l2==order(i),:)); % backward transformation
            [uf,sf,vf] = svd(T_for);
            [ub,sb,vb] = svd(T_back);
            
            T_rfor      = uf(:,1)*sf(1,1)*vf(:,1)';
            T_rback     = ub(:,1)*sb(1,1)*vb(:,1)';
            figure
            subplot(4,5,1)
            imagesc(G{order(i)});
            title(sprintf('G%d',order(i)));
            subplot(4,5,2)
            imagesc(G{order(i+1)});
            title(sprintf('G%d',order(i+1)));
            subplot(4,5,3)
            imagesc(predictGfromTransform(G{order(i)},T_rfor));
            title(sprintf('predicted G%d',order(i+1)));
            subplot(4,5,4)
            imagesc(predictGfromTransform(G{order(i)},T_rback));
            title(sprintf('predicted G%d',order(i)));
            
            subplot(4,5,6)
            imagesc(T_for);
            title(sprintf('forward full T layers %d-%d',order(i),order(i+1)));
            subplot(4,5,7)
            imagesc(T_back);
            title(sprintf('back full T layers %d-%d',order(i+1),order(i)));
            subplot(4,5,8)
            imagesc(T_rfor);
            title(sprintf('forward reduced T layers %d-%d',order(i),order(i+1)));
            subplot(4,5,9)
            imagesc(T_rback);
            title(sprintf('back reduced T layers %d-%d',order(i+1),order(i)));
            subplot(4,5,11)
            hist(diag(T_for),50);
            title(sprintf('range diagonal: %2.1f',range(diag(T_for))));
            subplot(4,5,12)
            hist(diag(T_back),50);
            title(sprintf('range diagonal: %2.1f',range(diag(T_back))));
            subplot(4,5,13)
            hist(diag(T_rfor),50);
            title(sprintf('range diagonal: %2.1f',range(diag(T_rfor))));
            subplot(4,5,14)
            hist(diag(T_rback),50);
            title(sprintf('range diagonal: %2.1f',range(diag(T_rback))));
            subplot(4,5,16)
            hist(rsa_vectorizeRDM(T_for),200);
            title(sprintf('range off-diagonal: %2.1f',range(rsa_vectorizeRDM(T_for))));
            subplot(4,5,17)
            hist(rsa_vectorizeRDM(T_rfor),200);
            title(sprintf('range off-diagonal: %2.1f',range(rsa_vectorizeRDM(T_back))));
            subplot(4,5,18)
            hist(rsa_vectorizeRDM(T_rfor),200);
            title(sprintf('range off-diagonal: %2.1f',range(rsa_vectorizeRDM(T_rfor))));
            subplot(4,5,19)
            hist(rsa_vectorizeRDM(T_rback),200);
            title(sprintf('range off-diagonal: %2.1f',range(rsa_vectorizeRDM(T_rback))));
            % here plot the added dimensions
            subplot(4,5,[10,15]);
            plot(t.dimFit(t.l1==order(i)&t.l2==order(i+1),:),'linewidth',2);
            hold on;
            plot(t.dimFit(t.l1==order(i+1)&t.l2==order(i),:),'--','linewidth',2);
            legend({'forward','backward'},'Location','SouthEast');
        end
    case 'plot_metrics_trueOrder'
        % plot relationship between metrics given the true order (distance)
        var = 'diagRange';
        vararginoptions(varargin,{'var'}); % which variable to plot
        a=load(fullfile(baseDir,'alexnet_alpha'));
        t=a.alpha{3};
        % determine the true order
        trueOrdDist = zeros(size(t.l1));
        rangeOff = zeros(size(t.l1));
        for i=1:size(t.l1,1)
            trueOrdDist(i)  = find(order==t.l2(i))-find(order==t.l1(i));
            RDM             = rsa_squareIPMfull(t.T(i,:));
            rangeOff(i)     = range(rsa_vectorizeRDM(RDM'));
        end
        t.trueOrdDist   = trueOrdDist;
        t.rangeOff      = rangeOff;
        % how many variables to plot
        if iscell(var)
            nVar=size(var,2);
        else
            nVar=1;
            var={var};
        end
        % plot forward and backward
        figure
        for i=1:nVar
            subplot(nVar,2,(i-1)*2+1)
            plt.scatter(abs(t.trueOrdDist),t.(var{i}),'subset',t.trueOrdDist<0);
            xlabel('Number of steps removed between layers');
            ylabel(var{i});
            title('Forward');
            subplot(nVar,2,(i-1)*2+2)
            plt.scatter(abs(t.trueOrdDist),t.(var{i}),'subset',t.trueOrdDist>0);
            title('Backward');
            xlabel('N(steps between layers)');
            ylabel(var{i});
        end
     
    case 'cluster_undirected'           % DEPRECIATED
        %% estimate the clustering for a given metric (undirected)
        C       = varargin{1};
        var     = varargin{2}; % which variable to consider
        nAlpha  = size(C,2); % number of cells in the cell array 
        OO=[]; 
        for a=1:nAlpha
            for i=unique(C{a}.distType)'
                A=getrow(C{a},C{a}.distType==i);
                % create an adjacency matrix
                t = rsa_squareRDM(A.dist');
                % create a matrix of similarity 
                t2 = t./max(max(t));
                W = 1-t2;
                [cl,centr] = kmeans(W,8);
                dC = diff(cl);
                %l=linkage(centr,'ward','euclidean');
                %dendrogram(l);
                % determine accuracy
                id = (1:32)';
                idx=rem(id,4)>0;
                O.error = length(find(dC(idx(1:(end-1)))))/length(dC); % percent errors in cluster assignment
                O.cluster = cl';
                O.centr   = {centr};
                % save other info
                O.distType    = i;
                O.distName    = {var};
                O.alphaType   = a;
                OO=addstruct(OO,O);
            end
        end
        varargout{1}=OO;
    case 'cluster_directed'             % DEPRECIATED
        %% estimate the clustering for a given metric (directed)
        C       = varargin{1};
        var     = varargin{2}; % which variable to consider
        OO=[];
        for i=1:length(var)
            % create an adjacency matrix
            t = rsa_squareIPMfull(C.(var{i})');
            % create a matrix of similarity - undirected now
            tu = t+t'; % two options: t+t' or t'*t+t*t'
            t2 = t./max(max(tu));
            W = 1-t2;
            [cl,centr] = kmeans(W,8);
            dC = diff(cl);
            %l=linkage(centr,'ward','euclidean');
            %dendrogram(l);
            % determine accuracy
            id = (1:32)';
            idx=rem(id,4)>0;
            O.error = length(find(dC(idx(1:(end-1)))))/length(dC); % percent errors in cluster assignment
            O.cluster = cl';
            O.centr   = {centr};
            % save other info
            O.distType    = i;
            O.distName    = {var};
            OO=addstruct(OO,O);
        end
        varargout{1}=OO;
   
    case 'plot_cluster_undirected'      % DEPRECIATED
        % plot the order for undirected graph
        T=varargin{1}; % data structure
        distType={'univariate','multivariate'};
        alphaType={'correlation','cosine'};
        for d=1:length(distType)
            figure
            for m=1:length(alphaType)
                t   = getrow(T,T.distType==d & T.alphaType==m);
                l   = linkage(t.centr{:},'ward','euclidean');
                figure
                dendrogram(l);
                hold on;
                ylim=get(gca,'ylim');
                plot(1:8,ylim(1),'o','markersize',10);
                title(sprintf('%s %s - error %2.1f',distType{d},alphaType{m},t.error));
            end
        end
    case 'plot_cluster_directed'        % DEPRECIATED
        % plot the order for undirected graph
        T=varargin{1}; % data structure
        distType={'corDist','scaleFit','diagFit','diagRange','dimension'};
        for d=1:length(distType)
            figure
            t   = getrow(T,T.distType==d);
            l   = linkage(t.centr{:},'ward','euclidean');
            figure
            dendrogram(l);
            hold on;
            ylim=get(gca,'ylim');
            plot(1:8,ylim(1),'o','markersize',10);
            title(sprintf('%s - error %2.1f',distType{d},t.error));
        end
     
    case 'estimate_topology_alpha'
        % here estimate topology for different alpha metrics - univariate /
        % multivariate
        % some parameters
        n_dim   = 2; % number of dimensions to consider
        n_neigh = 2; % number of neighbours to consider
        dataType = {'univariate','multivariate'};
        metricType = {'correlation','cosine'};
        a = load(fullfile(baseDir,sprintf('alexnet_%s_alpha',actUse)),'alpha');
        A = a.alpha;
        figure; indx=1;
        for d = 1:2 % univariate or multivariate
            for m = 1:2 % cosine or correlation
                % test on multi-cosine distance
                D = rsa_squareRDM(A{d}.dist(A{d}.distType==m)');
                % submit to topology function
                [mX,mp] = topology_estimate(D,n_dim,n_neigh,'dataType','raw');
                subplot(2,2,indx)
                hold on;
                W = full(mp.D);
                [r,c,val] = find(W);
                val = val./max(val); % renormalize
                for i=1:length(r)
                    plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',(1/val(i)),'Color',repmat(val(i),3,1)./(max(val)+0.1));
                end
                scatterplot(mX(:,1),mX(:,2),'label',(1:8),'split',(1:8)','markercolor',mColor,'markertype','.','markersize',40);
                title(sprintf('%s - %s',dataType{d},metricType{m}));
                indx=indx+1;
                
            end
        end
    case 'estimate_topology_directed'
        % estimate topology for directional metrics
        n_dim   = 2; % number of dimensions to consider
        n_neigh = 2; % number of neighbours to consider
        a = load(fullfile(baseDir,sprintf('alexnet_%s_alpha',actUse)),'alpha');
        A = a.alpha{3};
        metrics={'scaleDist','diagRange','eigStd','dimension'};
        for m = 1:length(metrics)
            t = rsa_squareIPMfull(A.(metrics{m})'); % t+t' to make it undirected
            [mX,mp] = topology_estimate(t+t',n_dim,n_neigh,'dataType','raw');
            subplot(1,length(metrics),m)
            hold on;
                W = full(mp.D);
                [r,c,val] = find(W);
                for i=1:length(r)
                    plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',(1/val(i)),'Color',repmat(val(i),3,1)./(max(val)+0.05));
                end
            scatterplot(mX(:,1),mX(:,2),'label',(1:8),'split',(1:8)','markercolor',mColor,'markertype','.','markersize',40);
            title(metrics{m});
        end
    case 'estimate_topology_firstLevel'
        % estimate topology using first level metrics instead of distances
         n_dim   = 2; % number of dimensions to consider
         n_neigh = 2; % number of neighbours to consider
         load(fullfile(baseDir,sprintf('alexnet_%s_RDM',actUse))); 
         load(fullfile(baseDir,sprintf('alexnet_%s_univariate',actUse)));
         load(fullfile(baseDir,sprintf('alexnet_%s_G',actUse)));
         % reshape the G
         Gn = zeros(numLayer,size(rsa_vectorizeIPMfull(G{1})',1));
         for i=1:numLayer
             Gn(i,:)=rsa_vectorizeIPMfull(G{i})';
         end
         [mR,~] = topology_estimate(RDM,n_dim,n_neigh,'dataType','raw');
         [mU,~] = topology_estimate(U,n_dim,n_neigh,'dataType','raw');
         [mG,~] = topology_estimate(Gn,n_dim,n_neigh,'dataType','raw');
         figure
         subplot(131)
         scatterplot(mU(:,1),mU(:,2),'label',1:8,'split',(1:8)','markercolor',mColor,'markertype','.','markersize',25);
         title('estimate from univariate activation');
         subplot(132)
         scatterplot(mR(:,1),mR(:,2),'label',1:8,'split',(1:8)','markercolor',mColor,'markertype','.','markersize',25);
         title('estimate from RDM');
         subplot(133)
         scatterplot(mG(:,1),mG(:,2),'label',1:8,'split',(1:8)','markercolor',mColor,'markertype','.','markersize',25);
         title('estimate from G');
    case 'estimate_topology_allUnits_subset'
        load(fullfile(baseDir,'imageAct_subsets'));
        numVox = 100;
        nNeigh = 20;
        nDim = 3;
        U = []; 
        for i=1:numLayer
            t = (act{i}(:,sample_wor(1:size(act{i},2),1,numVox)))';
            U = [U; t]; % univariate
        end
        D = squareform(pdist(U,'cosine'));
        ind = kron(1:numLayer,ones(1,numVox))';
        [m,mp] = topology_estimate(D,nDim,nNeigh,'dataType','raw');
        figure
        subplot(221)
        imagesc(mp.D); title('sorted adjacency matrix');
        subplot(222)
        imagesc(mp.DD); title('shortest path matrix');
        subplot(2,2,3:4);
        legLab = {'layer1','layer2','layer3','layer4','layer5','layer6','layer7','layer8'};
        scatterplot3(m(:,1),m(:,2),m(:,3),'split',ind,'markercolor',mColor,'markertype',{'.'},'markerfill',[1 1 1],'markersize',20,'leg',legLab);
        title(sprintf('estimated topology in %1.0fD with %2.0f neighbours per point',nDim,nNeigh)); axis equal; axis off;
    case 'allUnits_activation'
        % here determine the activation profile, similarity matrix of
        % all units (well, subset - 500 per layer)
        % univariate only (there is no multivariate per unit)
        load(fullfile(baseDir,'imageAct_subsets'));
        % reshape
        U = [];
        for i=1:numLayer
            U = [U; act{i}'];
        end
        for i=1:2
            if strcmp(aType{i},'correlation')
                    % additional step for correlation - first remove the mean
                    U  = bsxfun(@minus,U,mean(U,2));
            end
            U  = normalizeX(U);
            tmpR  = U*U'; % correlation across RDMs
            dist{i} = 1-tmpR;
        end
        save(fullfile(baseDir,'alexnet_allUnits_uniDist'),'dist');
    case 'allUnits_cluster'
        % determine clustering across units
        load(fullfile(baseDir,'alexnet_allUnits_uniDist'));
        
        thres = [0,500:500:4000];
        
        figure
        for a=1:size(dist,2)
            % create an adjacency matrix
            t = dist{a};
            % create a matrix of similarity
            t2 = t./max(max(t));
            W = 1-t2;
            subplot(2,2,a)
            imagesc(W);
%             hold on;
%             for i=1:numLayer
%                 drawline(thres(i)+1:thres(i+1),'dir','vert','lim',[thres(i) thres(i)+1],'color',[1 1 1]);
%                 drawline(thres(i)+1:thres(i+1),'dir','vert','lim',[thres(i+1) thres(i+1)+1],'color',[1 1 1]);
%                 drawline(thres(i)+1:thres(i+1),'dir','horz','lim',[thres(i) thres(i)+1],'color',[1 1 1]);
%                 drawline(thres(i)+1:thres(i+1),'dir','horz','lim',[thres(i+1) thres(i+1)+1],'color',[1 1 1])
%                 drawline(thres(i)+1:thres(i+1),'dir','horz','lim',[thres(i) thres(i)+1],'color',[1 1 1])
%             end
            colormap('hot');
            title(sprintf('%s univariate similarity matrix',aType{a}));
            [cl,~] = kmeans(W,8);
            subplot(2,2,a+2)
            imagesc([cl, kron(1:numLayer,ones(1,500))']); 
            title('estimated clusters (left) true layers (right)');
        end
        
    case 'HOUSEKEEPING:correctOrder'
        % reorder activation units
        activations_correct = cell(8,1);
        for i=1:size(actCorrect,1)
            activations_correct{i} = act{order(i)};
        end
        % save
        save(fullfile(baseDir,'imageActivations_alexNet_4Eva'),'activations_rand','activations_correct');
    case 'HOUSEKEEPING:subsets' 
        % split the original alexnet activations into smaller subsets
        % (ROIs) of 500 voxels each
        % to estimate if clustering will recover the correct layers
        % save as new structure
        nVox = 500; % 500 voxels in each ROI
        act_subsets = cell(numLayer,1);
        for i=1:numLayer
            act_subsets{i} = act{i}(:,sample_wor(1:size(act{i},2),1,nVox));
        end
        save(fullfile(baseDir,'imageAct_subsets'),'act_subsets');
    
        
    otherwise
        fprintf('This case does not exist!');
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
