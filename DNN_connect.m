function varargout = DNN_connect(what,varargin)
%function varargout = DNN_connect(what,varargin)
%calculates connectivity between layers of DNN (resnet50, vgg16)

baseDir = '/Users/Eva/Documents/Data/rsa_connectivity';
aType = {'correlation','cosine'};
dType = {'univariate','multivariate'};

switch what
    case 'firstLevel:wholePattern'
        % use this if estimating first level on whole patterns
        % otherwise submit the act to calculate function directly
        DNNname = 'alexnet'; % alexnet, resnet50, vgg16, alexnet_imagenet
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        [U,RDM,RDM_sqrt,~,~,G,~]=DNN_connect('firstLevel:calculate',act,size(act{1},1));
        save(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname)),'U','RDM','RDM_sqrt','G');
        varargout={U,RDM,RDM_sqrt,G};
    case 'firstLevel:calculate'
        % calculate first level estimates: U, RDM, G (cRDM, cG)
        Data = varargin{1};
        nCond = varargin{2};
        % initialize
        nPart = size(Data{1},1)/nCond;
        numLayer = size(Data,1);
        partVec = kron((1:nPart)',ones(nCond,1));
        condVec = kron(ones(nPart,1),(1:nCond)');
        G = cell(numLayer,1);
        cG = G;
        % contrast matrix for G->distances
        C = indicatorMatrix('allpairs',1:nCond);
        X = indicatorMatrix('identity_p',condVec);
        H = eye(nCond)-ones(nCond)./nCond;
        RDM = zeros(numLayer,size(C,1));
        cRDM = RDM; RDM_sqrt = RDM; cRDM_sqrt = RDM; % crossval, square root
        U = zeros(numLayer,nCond);
     %   fprintf('First level calculating layer:\n');
        for i=1:numLayer
       %     fprintf('%d.',i);
            nVox = size(Data{i},2);
            t=Data{i};
            for j=1:nPart % first remove the mean of each run
                t(partVec==j,:)=bsxfun(@minus,Data{i}(partVec==j,:),mean(Data{i}(partVec==j,:),1));
            end
            D               = pinv(X)*Data{i};
            G{i}            = D*D'/nVox;
            G{i}            = H*G{i}*H';
            RDM(i,:)        = diag(C*G{i}*C')';
            RDM_sqrt(i,:)   = ssqrt(RDM(i,:));
            D               = pinv(X)*t;
            U(i,:)          = mean(D,2)';
            cG{i}           = pcm_estGCrossval(Data{i},partVec,condVec);
            cG{i}           = cG{i}./trace(cG{i});
            cRDM(i,:)       = diag(C*cG{i}*C')';
            cRDM_sqrt(i,:)  = ssqrt(cRDM(i,:));
        end
    %    fprintf('...done all.\n');
        varargout{1}=U;
        varargout{2}=RDM;
        varargout{3}=RDM_sqrt;
        varargout{4}=cRDM;
        varargout{5}=cRDM_sqrt;
        varargout{6}=G;
        varargout{7}=cG;
    case 'firstLevel:normalize_overall'
        % transform activation units to z-scores
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        act_z = act;
        for i=1:nLayer
            a=act{i};
            [Y,I] = sort(a(:)); 
           % q = Y./length(Y) - 1/(2*length(Y));
            q = (1:length(Y))./length(Y) - 1/(2*length(Y));
            z = norminv(q,0,1);
            act_z{i} = reshape(z(I),size(a)); % reorder & reshape
        end
        act = act_z;
        dirZ = fullfile(baseDir,sprintf('%s_zscore',DNNname));
        dircheck(dirZ);
        save(fullfile(dirZ,sprintf('%s_zscore_activations',DNNname)),'act');
    case 'firstLevel:normalize_perUnit'
        % normalize activation units (per unit responses to stimuli)
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        for i=1:nLayer
            for j=1:size(act{i},2)
                [Y,~] = sort(act{i}(:,j));
                % q = Y./length(Y) - 1/(2*length(Y));
                q = (1:length(Y))./length(Y) - 1/(2*length(Y));
                z = norminv(q,0,1);
                act{i}(:,j) = z;
            end
            fprintf('Done: layer-%d\n',i);
        end
        dirZ = fullfile(baseDir,sprintf('%s_normUnit',DNNname));
        dircheck(dirZ);
        save(fullfile(dirZ,sprintf('%s_normUnit_activations',DNNname)),'act');
    case 'firstLevel:positive'
        % only keep the positive weights, set the rest to 0
        % mimics the rectified linear unit in the networks
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        for i=1:nLayer
            a=act{i};
            a(a<=0)=0;
            act{i}=a;
        end
        dirP = fullfile(baseDir,sprintf('%s_positive',DNNname));
        dircheck(dirP);
        save(fullfile(dirP,sprintf('%s_positive_activations',DNNname)),'act','-v7.3');
    case 'firstLevel:plot_act'
        % plot activation for each layer - per unit res
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        nFig = ceil(nLayer/8); % how many figures needed
        
        idx = 0; % count number of layers
        for i=1:nFig
            idxF = 1; % for figure
            figure
            for j=1:8 % arbitrary number of layers plotted
                subplot(2,4,idxF)
                histogram(act{idx+1}(:));
                title(sprintf('layer-%d',idx+1));
                idxF = idxF+1;
                idx=idx+1;
            end
        end
    case 'firstLevel:plot_dist'
        % plot distances for each layer (squared or square root)
        DNNname = 'alexnet';
        metric = 'RDM'; % RDM or RDM_sqrt
        vararginoptions(varargin,{'DNNname','metric'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname))); % load in act
        nLayer = size(RDM,1);
        nFig = ceil(nLayer/8); % how many figures needed
        
        idx = 0; % count number of layers
        D = eval(metric);
        for i=1:nFig
            idxF = 1; % for figure
            for j=1:8 % arbitrary number of layers plotted
                figure(i)
                subplot(2,4,idxF)
                histogram(D(idx+1,:));
                title(sprintf('layer-%d',idx+1));
                figure(10*i)
                subplot(2,4,idxF);
                imagesc(rsa_squareRDM(D(idx+1,:))); colorbar;
                title(sprintf('layer-%d',idx+1));
                idxF = idxF+1;
                idx=idx+1;
            end
        end
    case 'firstLevel:hist_dist'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname))); % load in act
       
        figure
        subplot(131)
        scatterplot(RDM(1,:)',RDM(2,:)'); xlabel('layer-1'); ylabel('layer-2');
        subplot(132)
        scatterplot(RDM(1,:)',RDM(3,:)'); xlabel('layer-1'); ylabel('layer-3');
        subplot(133)
        scatterplot(RDM(2,:)',RDM(3,:)'); xlabel('layer-2'); ylabel('layer-3');
        
    case 'secondLevel:calcConnect'
        % gets the second level metrics
        % second level: univariate / RDM connect, transformation of Gs
        % saves the outputs
        DNNname = 'vgg16';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname)));
        % estimate second level metrics (between RDMs, Gs)
        % 1) calculate distance based on mean activity (univariate)
        alpha{1} = DNN_connect('secondLevel:estimate_distance',U,'univariate');
        % 2) calculate distance between RDMs (cosine, correlation)
        alpha{2} = DNN_connect('secondLevel:estimate_distance',RDM,'RDM'); 
        % 3) calculate distance between RDMs (cosine, correlation) - sqrt
        alpha{3} = DNN_connect('secondLevel:estimate_distance',RDM_sqrt,'RDM'); 
        % 4) calculate transformation matrices T between Gs
        alpha{4} = DNN_connect('secondLevel:transformG',G);
        varargout{1}=alpha;
        % save the variables
        save(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
    case 'secondLevel:estimate_distance' 
        % esimate distance between layers - 2nd level
        RDM = varargin{1};
        dataType = varargin{2}; % univariate or multivariate connectivity
        alpha=[];
        for i=1:2
            D = DNN_connect('secondLevel:calcDist',RDM,aType{i});
            D.distType = ones(size(D.l1))*i;
            D.distName = repmat(aType(i),size(D.l1));
            alpha = addstruct(alpha,D);
        end
        varargout{1}=alpha;
        fprintf('Done estimating undirected distance metrics: %s.\n',dataType); 
    case 'secondLevel:calcDist'
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
    case 'secondLevel:transformG'
        % here estimate T + derive all metrics
        regularize=0;
        G=varargin{1};
        vararginoptions(varargin(2:end),{'regularize'});
        numG = max([size(G,1) size(G,2)]);
        if regularize
            for i=1:numG
                G{i}=G{i}+eye(size(G{i}));
            end
        end
        TT=[];
        for i1=1:numG % all pairs (because T is not symmetric)
            for i2=1:numG
                T.l1=i1;
                T.l2=i2;
                [Tr,~,~,~]              = calcTransformG(G{T.l1},G{T.l2}); % can retrieve cosine distance
                T.T                     = rsa_vectorizeIPMfull(Tr);
                T.distType              = ones(size(T.l1));
                % here characterise T
                % scaling
                scalar                  = mean(diag(Tr)); % which scalar
                scaleTr                 = eye(size(Tr))*scalar; % simplified transform - only scalar
                [~,~,T.scaleDist]       = predictGfromTransform(G{T.l1},scaleTr,'G2',G{T.l2});
                % diagonal + rangei1
                diagTr                  = Tr.*eye(size(Tr));
                [~,~,T.diagDist]        = predictGfromTransform(G{T.l1},diagTr,'G2',G{T.l2});
                T.diagRange             = max(diag(Tr))-min(diag(Tr));
                T.diagStd               = std(diag(Tr));
                % eigenvalues
                [~,T.eigRange,T.eigStd,T.eigComp,~] = eig_complexity(Tr);
                TT=addstruct(TT,T);
            end
        end
        varargout{1}=TT;
    case 'secondLevel:plotRDM'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname))); % load in act
        
        a=alpha{2}; 
        figure
        subplot(121)
        imagesc(rsa_squareRDM(a.dist(a.distType==1)')); title('corr-RDM'); colorbar;
        subplot(122)
        imagesc(rsa_squareRDM(a.dist(a.distType==2)')); title('cos-RDM'); colorbar;
        colormap hot;
    case 'isomap:undirected'
        n_dim   = 2; % number of dimensions to consider
        n_neigh = 3; % number of neighbours to consider
        DNNname = 'vgg16';
        vararginoptions(varargin,{'n_dim','n_neigh','DNNname'});

        a = load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
        A = a.alpha;
        numLayer = length(unique([A{1}.l1;A{1}.l2]));
        mColor = cell(1,numLayer);
        mCol = hsv(numLayer);
        for i=1:numLayer
            mColor{i}=mCol(i,:);
        end
        trueOrder = squareform(pdist((1:numLayer)'));
        for d = 1:2 % univariate or multivariate
            for m = 1:2 % cosine or correlation
                D = rsa_squareRDM(A{d}.dist(A{d}.distType==m)');
                [mX,mp] = topology_estimate(D,n_dim,n_neigh); % submit to topology function
                figure
                subplot(1,2,1)
                imagesc(D); colormap hot;
                tauAll = corr(rsa_vectorizeRDM(D)',rsa_vectorizeRDM(trueOrder)','Type','Kendall');
                title(sprintf('%s - %s, tau: %0.3f',dType{d},aType{m},tauAll));
                subplot(1,2,2)
                hold on;
                W = full(mp.D);
                [r,c,val] = find(W);
                val = val./max(val); % renormalize
                for i=1:length(r)
                  %  plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',(1/val(i)),'Color',repmat(val(i),3,1)./(max(val)+0.1));
                    plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
                end
               scatterplot(mX(:,1),mX(:,2),'label',(1:numLayer),'split',(1:numLayer)','markercolor',mColor,'markertype','.','markersize',10); 
               title(sprintf('%s - %s, isomap',dType{d},aType{m}));
            end
        end
    case 'isomap:directed'
        n_dim   = 2; % number of dimensions to consider
        n_neigh = 3; % number of neighbours to consider
        DNNname = 'vgg16';
        vararginoptions(varargin,{'n_dim','n_neigh','DNNname'});
        a = load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
        A = a.alpha{3};
        metrics={'scaleDist','eigComp'};
        
        numLayer = length(unique([A.l1;A.l2]));
        mColor = cell(1,numLayer);
        mCol = hsv(numLayer);
        for i=1:numLayer
            mColor{i}=mCol(i,:);
        end
        trueOrder = squareform(pdist((1:numLayer)'));
        for m = 1:length(metrics)
            t = rsa_squareIPMfull(A.(metrics{m})'); % t+t' to make it undirected
            D = t+t'; % symmetrize
            [mX,mp] = topology_estimate(t+t',n_dim,n_neigh);
            figure
            subplot(121)
            imagesc(D); colormap hot;
            tauAll = corr(rsa_vectorizeRDM(D)',rsa_vectorizeRDM(trueOrder)','Type','Kendall');
            title(sprintf('%s, tau: %0.3f',metrics{m},tauAll));
            subplot(122)
            hold on;
            W = full(mp.D);
            [r,c,val] = find(W);
            val = val./max(val); % renormalize
            for i=1:length(r)
                plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
            end
            scatterplot(mX(:,1),mX(:,2),'label',(1:numLayer),'split',(1:numLayer)','markercolor',mColor,'markertype','.','markersize',10);
            title(sprintf('%s, isomap',metrics{m}));
        end
        
    case 'noise:constructTruth'
        % construct noiseless ground truth for RDMs
        DNNname = 'alexnet';
        dataInd = [1 1 2 2 3 3 4 4 4]; % which alpha to consider
        alphaInd = [1 2 1 2 1 2 3 4 5];
        dirMetrics  = {'scaleDist','diagDist','eigComp'};
        vararginoptions(varargin,{'DNNname'});
        % load
        load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
        A = alpha;
        DD = [];
        for i=1:length(dataInd)
            d = dataInd(i);
            a = alphaInd(i);
            if d<4 % undirected
                D.dist = A{d}.dist(A{d}.distType==a)';
            else % directed
                dist = rsa_squareIPMfull(A{d}.(dirMetrics{a-2})');
                D.dist = rsa_vectorizeRDM(dist+dist');
            end
            D.alphaInd  = a;
            D.dataInd   = d;
            D.indx      = i;
            DD = addstruct(DD,D);
        end
        save(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname)),'-struct','DD');
    case 'noise:simulate'
        nPart       = 8;
        nSim        = 25;
        nUnits      = 500;
        noiseType   = 'within_low'; % allEqual or neighbours
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname'});
        % initialize
        dataInd     = [1 1 2 2 4 4 3 3 3 3 3 3 3 5 5 5 5 5 5 5 6]; % anzelotti as 6
        alphaInd    = [1 2 1 2 1 2 3 4 5 6 7 8 9 3 4 5 6 7 8 9 10];
        RDMInd      = [1 2 3 4 3 4 5 6 7 8 9 10 11 5 6 7 8 9 10 11 0];
        dirMetrics  = {'scaleDist','diagDist','eigComp'};
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        trueOrder = squareform(pdist((1:nLayer)'));
        switch noiseType
            case 'allEqual'
                varReg = [0.01,0.1,0.5,2,5,10];
                corrReg = 0:0.2:0.8;
            case 'neighbours'
                error('Not implemented!');
            case 'within'
               % varReg = [0,0.01,0.1,0.5,1,2:1:15];
                 varReg = [0,0.1,0.5,1:1:10];
                 corrReg = 0;
            case 'within_low'
              %  varReg = [0,1.7:0.1:3.5];
                varReg = [0,1.7:0.2:3.4];
                corrReg = 0;
        end
        VV=[];
        for v=varReg        % within noise
            if v==0
                corR=0;
            else
                corR=corrReg;
            end
            for r=corR         % correlated noise 
                for n=1:nSim    % number of simulations
                    % here subsample + repeat the true pattern for each partition
                    act_subsets = cell(nLayer,1);
                    data = cell(nLayer,1);
                    for i=1:nLayer
                        rUnits = randperm(size(act{i},2)); % randomise the order of units
                        act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                        data{i} = repmat(act_subsets{i},nPart,1);
                    end
                    tElapsed=tic;
                    Data = addSharedNoise(data,v,r,noiseType);
                    RDMconsist = rdmConsist(Data,nPart,size(act_subsets{1},1));
                    [fD{1},fD{2},fD{3},fD{4},fD{5}]=DNN_connect('firstLevel:calculate',Data,nCond); % order: uni, RDM, cRDM, G, cG
                    A{1} = DNN_connect('secondLevel:transformG',fD{3});
                    A{2} = DNN_connect('secondLevel:transformG',fD{5});
                    % cycle around all combinations of data / metrics
                    for i=1:length(dataInd)
                        if ismember(dataInd(i),[1,2,4])
                            t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                            T = rsa_squareRDM(t.dist');
                            V.corrNoiseless = corr(TD.dist{RDMInd(i)},rsa_vectorizeRDM(T)');
                        elseif dataInd(i)==3
                            T = rsa_squareIPMfull(A{dataInd(i)-2}.(dirMetrics{alphaInd(i)-2})');
                            V.corrNoiseless = corr(TD.dist{RDMInd(i)},T(:));
                            T = T+T'; % symmetrize
                        elseif dataInd(i)==5
                            T = rsa_squareIPMfull(A{dataInd(i)-3}.(dirMetrics{alphaInd(i)-2})');
                            V.corrNoiseless = corr(TD.dist{RDMInd(i)},T(:));
                            T = T+T'; % symmetrize
                        else % Anzellotti
                            T = anzellottiDist(Data,nPart,size(act{1},1));       
                            V.corrNoiseless = NaN;
                        end
                        [~,sortIdx]=sort(T,2);
                        sortIdx = sortIdx - 1;
                        V.tauTrue = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(sortIdx)');
                        V.RDM           = rsa_vectorizeRDM(T);
                        V.RDMconsist    = RDMconsist;                 
                        V.dataType      = dataInd(i); % uni / multi / directional
                        V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                        V.metricIndex   = i;
                        V.numSim        = n;
                        V.corrReg       = r;
                        V.varReg        = v;
                        VV=addstruct(VV,V);
                    end
                    fprintf('%d. ',n);
                    toc(tElapsed);
                end
                fprintf('\nFinished variance %d.\n\n',v);
            end
            fprintf('\nFinished correlation %d.\n\n',r);
        end
        save(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)),'-struct','VV');
        fprintf('\nDone simulations %s: %s \n',DNNname,noiseType);
    case 'noise:plot'
        noiseType = 'within_low';
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        
        for i=unique(T.dataType)'
            figure
            if i~=6
                subplot(211)
                plt.line(T.varReg,T.corrNoiseless,'split',T.metricType,'subset',T.dataType==i);
                xlabel('Noise'); ylabel('Comparison to noiseless');
            end
            subplot(212)
            plt.line(T.varReg,T.tauTrue,'split',T.metricType,'subset',T.dataType==i);
            xlabel('Noise'); ylabel('Comparison to true structure (rank)');
        end
        
    case 'noiseless:subsetCond'
        % validate topology in the noiseless case by subsampling conditions
        % use the normalized activation units here
       % validate topology in the noiseless case by subsampling units
        nSim        = 100; % number of simulations
        nCond       = 50; % number of units to sample at a time
        DNNname     = 'alexnet';
        dirMetrics  = {'scaleDist','diagDist','eigComp'}; % directional metrics to consider
        vararginoptions(varargin,{'nSim','nUnits','nDim','nNeigh','DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        numLayer = size(act,1);
        order = 1:numLayer; trueOrder = squareform(pdist(order'));
        
        dataType    = [1 1 2 2 3 3 repmat(4,1,length(dirMetrics))]; % uni / RDM / RDM_sqrt / directional
        metricType  = [1 2 1 2 1 2 3:2+length(dirMetrics)]; % corr / cos / scaleDist...
        VV = [];
        for n=1:nSim % simulations
            tStart=tic;
            rCond = randperm(size(act{1},1)); % randomise the conditions
            act_subsets = cell(numLayer,1);
            for i=1:numLayer
                act_subsets{i} = act{i}(rCond(1:nCond),:);
            end
            %% 1) here estimate first level
            [fD{1},fD{2},fD{3},~,~,fD{4},~]=DNN_connect('firstLevel:calculate',act_subsets,size(act_subsets{1},1)); % order: uni, RDM, cRDM, G, cG            
            %% 2) estimate second level metrics (between RDMs, Gs)
            % calculate transformation matrices T between Gs
            A = DNN_connect('secondLevel:transformG',fD{4});
            %% 3) estimate topology - neighbourhood, then relate to true order
            for m = 1:length(metricType)
                d = dataType(m);
                if d<4 % uni / multivariate - corr / cos
                    t = DNN_connect('secondLevel:calcDist',fD{dataType(m)},aType{metricType(m)});
                    D = rsa_squareRDM(t.dist');
                else % directional
                    t = rsa_squareIPMfull(A.(dirMetrics{m-6})');
                    D = t+t'; % make it symmetric
                end 
                V.corrNoiseless = corr(TD.dist(m,:)',rsa_vectorizeRDM(D)');
                NN              = construct_neighbourhood(D);
                V.tauTrue       = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)');
                V.metric        = m;
                V.dataType      = d; % uni / multi / directional
                V.metricType    = metricType(m); % corr / cos / eigStd ...
                V.numSim        = n;
                VV=addstruct(VV,V);
            end
            fprintf('%d. simulation done...',n);
            toc(tStart);
        end
        % save here
        save(fullfile(baseDir,DNNname,'validate_noiseless_subsetCond'),'-struct','VV'); 
        fprintf('Done subsampling conditions: %s\n\n',DNNname);
    case 'noiseless:subsetUnit'
       % validate topology in the noiseless case by subsampling units
        nSim        = 100; % number of simulations
        nUnits      = 500; % number of units to sample at a time
        DNNname     = 'alexnet';
        dirMetrics  = {'scaleDist','diagDist','eigComp'}; % directional metrics to consider
        vararginoptions(varargin,{'nSim','nUnits','nDim','nNeigh','DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        numLayer = size(act,1);
        order = 1:numLayer; trueOrder = squareform(pdist(order'));
        
        dataType    = [1 1 2 2 3 3 repmat(4,1,length(dirMetrics))]; % uni / RDM / RDM_sqrt / directional
        metricType  = [1 2 1 2 1 2 3:2+length(dirMetrics)]; % corr / cos / scaleDist...
        VV = [];
        for n=1:nSim % simulations
            tStart=tic;
            act_subsets = cell(numLayer,1);
            for i=1:numLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
            end
            %% 1) here estimate first level
            [fD{1},fD{2},fD{3},~,~,fD{4},~]=DNN_connect('firstLevel:calculate',act_subsets,size(act_subsets{1},1)); % order: uni, RDM, cRDM, G, cG            
            %% 2) estimate second level metrics (between RDMs, Gs)
            % calculate transformation matrices T between Gs
            A = DNN_connect('secondLevel:transformG',fD{4});
            %% 3) estimate topology - neighbourhood, then relate to true order
            for m = 1:length(metricType)
                d = dataType(m);
                if d<4 % uni / multivariate - corr / cos
                    t = DNN_connect('secondLevel:calcDist',fD{dataType(m)},aType{metricType(m)});
                    D = rsa_squareRDM(t.dist');
                else % directional
                    t = rsa_squareIPMfull(A.(dirMetrics{m-6})');
                    D = t+t'; % make it symmetric
                end 
                V.corrNoiseless = corr(TD.dist(m,:)',rsa_vectorizeRDM(D)');
                NN              = construct_neighbourhood(D);
                V.tauTrue       = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)');
                V.metric        = m;
                V.dataType      = d; % uni / multi / directional
                V.metricType    = metricType(m); % corr / cos / eigStd ...
                V.numSim        = n;
                VV=addstruct(VV,V);
            end
            fprintf('%d. simulation done...',n);
            toc(tStart);
        end
        % save here
        save(fullfile(baseDir,DNNname,'validate_noiseless_subsetUnit'),'-struct','VV'); 
        fprintf('Done subsampling units: %s\n\n',DNNname);
    case 'noiseless:plot_subset'
        subsetType = 'Cond'; % Cond or Units
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'subsetType','DNNname'});
        T = load(fullfile(baseDir,DNNname,sprintf('validate_noiseless_subset%s',subsetType)));
        tickLab = {'uni-corr','uni-cos','multi-corr','multi-cos',...
            'G-scaleDist','G-diagDist','G-diagRange',...
            'G-diagStd','G-eigRange','G-eigStd','G-eigComp'};
        figure
        subplot(211)
        barplot(T.dataType,T.tauTrue,'split',T.metricType);
        set(gca,'XTickLabel',tickLab); ylabel('Comparison to true ordering');
        title(sprintf('subsampling %s',subsetType));
        subplot(212)
        barplot(T.dataType,T.corrNoiseless,'split',T.metricType);
        set(gca,'XTickLabel',tickLab); ylabel('Comparison to complete noiseless');
    
    case 'eigDim'
        DNNname='vgg16';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)));
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname)),'G');
        numLayer=size(G,1);
        A = alpha{3};
        
        TT=[];DD=[]; 
        for j=1:numLayer % characterise per layer
            [~,D.eigRange,D.eigStd,D.eigComp,D.eigComp2] = eig_complexity(G{j});
            D.layer = j;
            DD=addstruct(DD,D);
        end
        
        for i=1:size(A.diagDist,1) % characterise per connection
            Tr = rsa_squareIPMfull(A.T(i,:));
            [~,T.eigRange,T.eigStd,T.eigComp,T.eigComp2] = eig_complexity(Tr);   
            T.l1=A.l1(i);
            T.l2=A.l2(i);
            T.geoMean = sqrt(DD.eigComp(DD.layer==A.l1(i))*DD.eigComp(DD.layer==A.l2(i)));
            T.geoMean2 = sqrt(DD.eigComp2(DD.layer==A.l1(i))*DD.eigComp2(DD.layer==A.l2(i)));
            TT=addstruct(TT,T);
        end

        figure
        subplot(221)
        plt.line(DD.layer,DD.eigComp2);
        xlabel('Layer'); title('eigComp'); ylabel('Gs')
        % plt.dot(abs(TT.l1-TT.l2),TT.eigComp2,'split',TT.l1<TT.l2,'subset',TT.l1~=TT.l2);
        subplot(222)
        plt.line(DD.layer,DD.eigComp);
        xlabel('Layer'); title('eigComp (dist norm)'); ylabel('');
        subplot(223)
        plt.line(abs(TT.l1-TT.l2),TT.eigComp2,'split',TT.l1<TT.l2,'subset',TT.l1~=TT.l2,...
            'leg',{'back','forw'},'leglocation','northwest');
           xlabel('Number of edges in-between'); ylabel('Connection');
        subplot(224)
        plt.line(abs(TT.l1-TT.l2),TT.eigComp,'split',TT.l1<TT.l2,'subset',TT.l1~=TT.l2,...
             'leg',{'back','forw'},'leglocation','northwest');
        xlabel('Number of edges in-between'); ylabel('');
  
        figure
        subplot(131)
        plt.line(TT.l2-TT.l1,TT.eigComp,'subset',ismember(TT.l1,1)&TT.l1~=TT.l2);
        xlabel('N(layer) from layer 1');
        ylabel('eigComp (normDist)');
        subplot(132)
        plt.line(TT.l1-TT.l2,TT.eigComp,'subset',ismember(TT.l1,8)&TT.l1~=TT.l2);
        xlabel('N(layer) from layer 8');
        ylabel('eigComp (normDist)');
        subplot(133)
        plt.line(TT.l1-TT.l2,TT.eigComp,'subset',ismember(TT.l1,numLayer)&TT.l1~=TT.l2);
        xlabel('N(layer) from last layer');
        ylabel('eigComp (normDist)');
        plt.match('y');
   
        nTrans = numLayer-1;
        for i=1:nTrans
            figure
            subplot(121)
            plt.bar(DD.layer,DD.eigComp2,'subset',ismember(DD.layer,[i,i+1]));
            title('Gs'); ylabel('eigenvalue complexity');
            subplot(122)
            t=getrow(TT,ismember(TT.l1,[i,i+1])&ismember(TT.l2,[i,i+1])&TT.l1~=TT.l2);
            plt.bar(t.l1,t.eigComp2,'split',t.l1,...
                'leg',{'forward','backward'},'leglocation','southeast');
            title(sprintf('Ts: %2.2f vs. %2.2f',t.eigComp2(1),t.eigComp2(2)));
            ylabel('eigenvalue complexity');
        end
        
    case 'HOUSEKEEPING:normalizeUnits'
        % here normalize the activation in units (so the variance of signal
        % is comparable across layers)
        act = varargin{1};
        numLayer = size(act,1);
        actN = cell(numLayer,1);
        for i=1:numLayer
            actN{i}=bsxfun(@minus,act{i},mean(act{i},1));  % first here remove the mean activation
            actN{i}=actN{i}./max(max(actN{i}));
        end
        varargout{1}=actN;      
    
    case 'run_job'
        DNN_connect('firstLevel:positive','DNNname','vgg16');
        DNN_connect('firstLevel:positive','DNNname','resnet50');
        DNN_connect('firstLevel:wholePattern','DNNname','vgg16_positive');
        DNN_connect('firstLevel:wholePattern','DNNname','resnet50_positive');
        DNN_connect('noise:constructTruth','DNNname','alexnet_positive');
        DNN_connect('noise:constructTruth','DNNname','alexnet_imagenet_positive');
        DNN_connect('noise:constructTruth','DNNname','alexnet_positive');
        DNN_connect('noise:constructTruth','DNNname','vgg16_positive');
        DNN_connect('noise:constructTruth','DNNname','vgg16');
        DNN_connect('noise:constructTruth','DNNname','resnet50');
        DNN_connect('noise:constructTruth','DNNname','resnet50_positive');
        DNN_connect('noiseless:subsetCond');
        DNN_connect('noiseless:subsetCond','DNNname','alexnet_positive');
        DNN_connect('noiseless:subsetCond','DNNname','alexnet_imagenet_positive');
        DNN_connect('noiseless:subsetCond','DNNname','alexnet_imagenet');
        DNN_connect('noiseless:subsetCond','DNNname','vgg16');
        DNN_connect('noiseless:subsetCond','DNNname','vgg16_positive');
        DNN_connect('noiseless:subsetCond','DNNname','resnet50');
        DNN_connect('noiseless:subsetCond','DNNname','resnet50_positive');
        DNN_connect('noiseless:subsetUnit');
        DNN_connect('noiseless:subsetUnit','DNNname','alexnet_positive');
        DNN_connect('noiseless:subsetUnit','DNNname','alexnet_imagenet_positive');
        DNN_connect('noiseless:subsetUnit','DNNname','alexnet_imagenet');
        DNN_connect('noiseless:subsetUnit','DNNname','vgg16');
        DNN_connect('noiseless:subsetUnit','DNNname','vgg16_positive');
        DNN_connect('noiseless:subsetUnit','DNNname','resnet50');
        DNN_connect('noiseless:subsetUnit','DNNname','resnet50_positive');
        
        
    otherwise 
        fprintf('no such case!\n');

end
end
%% Local functions
function dircheck(dir)
% Checks existance of specified directory. Makes it if it does not exist.

if ~exist(dir,'dir');
    %warning('%s didn''t exist, so this directory was created.\n',dir);
    mkdir(dir);
end
end
function [eigT,eigRange,eigStd,eigComp,eigComp2] = eig_complexity(M)
%function eigC = eig_complexity(M)
%calculates eigenvalues of matrix M
%and derivatives
eigT        = eig(M)';
eigT        = round(eigT,10); % to ensure that negative values are really neg, not 0
if any(eigT<0)
    eigT=eigT(eigT>0);
end
eigRange    = max(eigT)-min(eigT);
eigStd      = std(eigT);
eigComp     = 1-((sum(eigT))^2/(sum(eigT.^2)))/size(M,1);
eigComp2    = ((sum(eigT))^2/(sum(eigT.^2)));
end
function data               = addSharedNoise(data,Var,r,noiseType)
% input: 
% data - datasets
% alpha - variance
% r - correlation
% noiseType - allEqual or neighbours
    nDataset = size(data,1);
    nTrials = size(data{1},1);
    nVox    = size(data{1},2);
    
    % add shared noise (shared within / across regions)
    Z = normrnd(0,1,nTrials,nDataset*nVox);
    Z = Z./max(max(Z));
    switch noiseType
        case 'within'
            Zn = Z;
        case 'noiseless'
            Zn = Z;
        case 'within_low'
            Zn = Z;
        case 'allEqual'
            A = eye(nDataset)*Var+(ones(nDataset)-eye(nDataset))*Var*r;
            P = kron(A,eye(nVox));
            Zn = Z*sqrtm(P);     % shared noise matrix across reg
        case 'neighbours'
            % first structured shared noise
            kernelWidth = 500; % decide on the kernel width
            t           = 1:nVox*nDataset;
            dist        = pdist(t');
            noiseKernel = exp((-0.5*dist)/kernelWidth);
            P           = squareform(noiseKernel);
            %   Zn          = Z*real(sqrtm(P));
            Zn          = Z*P;
            % modulate the relative strength of the shared noise 
            Zn = Zn./max(max(Zn));
            Zn = Z.*(1-r)+(Zn.*r);
    end
    Zn = Zn./max(max(Zn));
    for i=1:nDataset
        data{i} = data{i} + Var.*Zn(:,(i-1)*nVox+1:i*nVox);
        data{i} = data{i}./max(max(data{i}));
    end
end
function RDMconsist         = rdmConsist(Data,nPart,nCond)
% function RDMconsist = rdmConsist(data,nPart,nCond);
% calculate the consistency of RDMs across runs
numLayer = size(Data,1);
consist_lay = zeros(numLayer,1);
partVec = kron((1:nPart)',ones(nCond,1));
% contrast matrix for G->distances
C = indicatorMatrix('allpairs',1:nCond);
X = indicatorMatrix('identity_p',[1:nCond]);
H = eye(nCond)-ones(nCond)./nCond;
for i=1:numLayer
    nVox = size(Data{i},2);
    % calculate mean activation
    t=Data{i};
    for j=1:nPart % first remove the mean of each run
        t(partVec==j,:)=bsxfun(@minus,Data{i}(partVec==j,:),mean(Data{i}(partVec==j,:),1));
        D           = pinv(X)*t(partVec==j,:);
        G           = D*D'/nVox;
        G           = H*G*H';
        RDM(j,:)    = diag(C*G*C')';
    end
    consist_lay(i)  = mean(rsa_vectorizeRDM(corr(RDM')));
end
RDMconsist = mean(consist_lay);
end
function R2_dist            = anzellottiDist(data,nPart,nCond)
% function dist = anzellottiDist(data,partVec,condVec)
% calculates a relationship between data of different regions
% for now the distance output is 1-R2 
nData   = size(data,1);
partVec = kron((1:nPart)',ones(nCond,1));          
condVec = kron(ones(nPart,1),(1:nCond)');
ind     = indicatorMatrix('allpairs',1:nData);
dist    = zeros(size(ind,1),1);
for i=1:size(ind,1)
    ind1 = find(ind(i,:)==1);
    ind2 = find(ind(i,:)==-1);
    dist(i) = multiDependVox(data{ind2},data{ind1},partVec,condVec,'type','reduceAB');
end
R2_dist=rsa_squareRDM(dist');
end
function NN                 = construct_neighbourhood(D)
% constructs the neighbourhood in the given distance matrix
nLayer = size(D,1);
[~,ind]=sort(D,2,'ascend');
NN = zeros(nLayer);
for i=1:nLayer
    for j=1:nLayer
        NN(i,j) = find(ind(i,:)==j);
    end
end
NN = NN-1;
end