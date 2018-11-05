function varargout = rsa_connect(what,varargin)

baseDir = '/Volumes/MotorControl/data/rsa_connectivity';
% example RDM distance matrix
%load(fullfile(baseDir,'RDM.mat'));

averageType={'allPart','averagePart','crossval'};
distLabels={'corr','cosine','euclidean','distcorr'};

% for plotting
gray=[120 120 120]/255;
lightgray=[170 170 170]/255;
silver=[230 230 230]/255;
black=[0 0 0]/255;
blue=[49,130,189]/255;
mediumblue=[128,231,207]/255;
lightblue=[158,202,225]/255;
red=[222,45,38]/255;
mediumred=[237,95,76]/255;
lightred=[252,146,114]/255;

sBW = style.custom({gray,lightgray,black,silver});
sCol  = style.custom({blue,mediumblue,mediumred,lightblue});
s2  = style.custom({gray,lightgray,lightred,silver});
styTrained_sess=style.custom({red,mediumred,lightred});
styUntrained_sess=style.custom({blue,mediumblue,lightblue});

switch(what)
    
    case 'simulateData'
        nCond       = 5;
        nVox        = 1000;
        nPart       = 8;
        nRegion     = 2;
        nSim        = 1000;
        signal      = 1;
        theta       = 1;
        noise       = [0:0.1:1];
        rdmType     = 'randomRDM'; % what type of simulation: randomRDM, sameRDM
        vararginoptions(varargin,{'theta','numCond','numVox','numPart','numDataset','numSim','rdmType','simuType','noise','signal'});
        
        NN=[]; 
        for n1=1:size(noise,2);
            for n2=1:size(noise,2)
                for s=1:nSim
                    % 1) generate data
                    S.numPart = nPart;
                    S.numVox  = nVox;
                        clear D;
                    for d=1:nRegion
                        D{d}    = randn(nCond);
                        if strcmp(rdmType,'sameRDM')
                            D{d} = D{1};
                        end
                        M{d}    = makeModel(rdmType,D{d},nCond); % make model for data generation
                        if d==1
                            nIdx = noise(n1);
                        else
                            nIdx = noise(n2);
                        end
                        [data(2),partVec,condVec] = pcm_generateData(M{d},theta,S,1,10,nIdx);
                        trueRDM(d,:)  = rsa_vectorizeRDM(M{d}.Ac); % trueRDM
                    end
                    
                    % 2) calculate true distance
                    N.trueDist = calcDist(trueRDM);
                    
                    % 3) evaluate
                    for rt = 1:length(averageType);
                        % 3a) how to calculate RDM - 1st level
                        switch averageType{rt}
                            case 'allPart'
                                calcRDM = makeRDM_allPart(data);
                            case 'averagePart'
                                calcRDM = makeRDM_average(data,condVec);
                            case 'crossval'
                                calcRDM = makeRDM_crossval(data,partVec,condVec);
                        end
                        % 3b) calculate distances of two datasets - 2nd level
                        N.calcDist = calcDist(calcRDM);
                        % dist type labelling
                        N.distType        = [1:4]';
                        N.distLabel(1,:)  = {'correlation'};
                        N.distLabel(2,:)  = {'cosine'};
                        N.distLabel(3,:)  = {'euclidean'};
                        N.distLabel(4,:)  = {'distCorr'};
                        % other info
                        N.numSim        = repmat(s,4,1);
                        N.noise1        = repmat(noise(n1),4,1);
                        N.noise2        = repmat(noise(n2),4,1);
                        N.averageType   = repmat(rt,4,1);
                        NN=addstruct(NN,N);
                    end; % rdmType
                end; % num simulation
                fprintf('Simulation done: \tnoise level: %2.1f-%2.1f \n',noise(n1),noise(n2));
            end % noise
        end
        % save the structure
        save(fullfile(baseDir,sprintf('simulations_%s',rdmType)),'-struct','NN');
    case 'simulateData_comb'
        nCond = 5;
        theta = 1;
        nPart = 8;
        nVox = 1000;
        nSim = 20;
        signal = 1;
        noise  = [0:0.1:1];
        vararginoptions(varargin,{'numCond','numPart','numVox','numSim','theta','corrInit','noise','rdmType'});
        
        % defaults for simulations
        S.numPart = nPart;
        S.numVox  = nVox;
        % set two random distance matrices
        clear D;
        D{1} = rand(nCond);
        D{2} = rand(nCond);
        % set diagonals to 0
        D{1}(eye(size(D{1}))==1)=0;
        D{2}(eye(size(D{2}))==1)=0;
        % weights for calculating the third RDM - combination of 1st, 2nd
        w1 = [1:-0.1:0];
        w2 = [0:0.1:1];
        
        NN = [];        
        for n = 1:size(noise,2);
            for i = 1:size(w1,2)
                D{3} = D{1}.*w1(i) + D{2}.*w2(i);
                for s = 1:nSim
                    %1) generate data from the three distance matrices
                    for d=1:3
                        M{d}    = makeModel('combRDM',D{d},nCond); % make model for data generation
                        [data(d),partVec,condVec] = pcm_generateData(M{d},theta,S,1,signal,noise(n));
                        trueRDM(d,:)  = rsa_vectorizeRDM(M{d}.Ac); % trueRDM
                    end
                    % 2) calculate true distance
                    N.trueDist = calcDist(trueRDM);
                    
                    for r = 1:length(averageType);
                        % 3) estimate RDM from data
                        % 3a) average data
                        switch averageType{r}
                            case 'allPart'
                                calcRDM = makeRDM_allPart(data);
                            case 'averagePart'
                                calcRDM = makeRDM_average(data,condVec);
                            case 'crossval'
                                calcRDM = makeRDM_crossval(data,partVec,condVec);
                        end
                        
                        % 3b) calculate distances of two datasets - 2nd level
                        N.calcDist = calcDist(calcRDM);
                        % dist type labelling
                        N.distType        = [1:4]';
                        N.distLabel(1,:)  = {'correlation'};
                        N.distLabel(2,:)  = {'cosine'};
                        N.distLabel(3,:)  = {'euclidean'};
                        N.distLabel(4,:)  = {'distCorr'};
                        % other info
                        N.numSim        = repmat(s,4,1);
                        N.noise1        = repmat(noise(n),4,1);
                        N.noise2        = repmat(noise(n),4,1);
                        N.weight1       = repmat(w1(i),4,1);
                        N.weight2       = repmat(w2(i),4,1);
                        N.averageType   = repmat(r,4,1);
                        NN=addstruct(NN,N);
                        
                    end; % rdmType
                end;  %simulation
                fprintf('%d.\n',i);
            end; %weighted combination
            fprintf('Simulation done: \tnoise level: %2.1f\n',noise(n));
        end; % noise levels
        
        save(fullfile(baseDir,'simulations_combRDM'),'-struct','NN');
        
    case 'evaluate'
      rdmType = 'randomRDM'; % what type of simulation: random, sameRDM
      vararginoptions(varargin,{'rdmType'});
      T = load(fullfile(baseDir,sprintf('simulations_%s',rdmType)));
      EE=[];
      for n1=unique(T.noise1)'
          for n2=unique(T.noise2)'
              R1=pivottable(T.averageType,T.distType,[T.trueDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R2=pivottable(T.averageType,T.distType,[T.calcDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R3=pivottable(T.averageType,T.distType,[T.calcDist-T.trueDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R4=pivottable(T.averageType,T.distType,[T.calcDist],'var','subset',T.noise1==n1 & T.noise2==n2);
              R5=pivottable(T.averageType,T.distType,[T.trueDist-T.calcDist],'sqrt(mean(x.^2))','subset',T.noise1==n1 & T.noise2==n2);
              
              E.trueDist    = R1(:);
              E.calcDist    = R2(:);
              E.bias        = R3(:);
              E.var         = R4(:);
              E.rmse        = R5(:);
              E.averageType = repmat([1:3]',4,1);
              E.distType    = kron([1:4],ones(1,3))';
              E.noise1      = repmat(n1,12,1);
              E.noise2      = repmat(n2,12,1);

              EE=addstruct(EE,E);
          end
      end
      
      save(fullfile(baseDir,sprintf('evaluate_%s',rdmType)),'-struct','EE');
   
    case 'noiseless'    
        nCond = 5;      
        signal = [1,5,10:10:100];
        theta = ones(nCond*(nCond-1)/2,1);
        noise = 0;
        nPart = 9;
        nVox = 1000;
        vararginoptions(varargin,{'nCond','signal','theta','noise'});

        D = randn(nCond);
        D(1:nCond+1:end)=zeros(nCond,1);
        NN=[]; 
        S.numPart = nPart;
        S.numVox  = nVox;
        trueRDM=repmat(rsa_vectorizeRDM(D),2,1);
        trueDist = calcDist(trueRDM);
        M=makeModel('sameRDM',D,nCond); 
        for s=signal
            [data(1),partVec,condVec] = pcm_generateData(M,M.theta,S,1,1,noise);
            [data(2),partVec,condVec] = pcm_generateData(M,M.theta,S,1,s,noise);
            N.calcDist = calcDist(trueRDM);
            calcRDM = makeRDM_crossval(data,partVec,condVec);
            N.calcDist = calcDist(calcRDM);
            % dist type labelling
            N.distType        = [1:4]';
            N.distLabel(1,:)  = {'correlation'};
            N.distLabel(2,:)  = {'cosine'};
            N.distLabel(3,:)  = {'euclidean'};
            N.distLabel(4,:)  = {'distCorr'};
            N.signal1 = repmat(1,4,1);
            N.signal2 = repmat(s,4,1);
            NN = addstruct(NN,N);
        end

        figure
        plt.line(NN.signal2,NN.calcDist,'split',NN.distType,'leg',{'correlation','cosine','euclidean','distCorr'},'leglocation','northwest');
        ylabel('distance between identical RDMs');
        xlabel('signal proportion of RDM 1 vs. 2');
    case 'noiseDiff'        
        nCond = 5;      
        signal = 1;
        theta = 1;
        noise = [0:0.1:1];
        nPart = 9;
        nVox = 1000;
        vararginoptions(varargin,{'nCond','signal','theta','noise'});

        D = randn(nCond);
        NN=[]; 
        S.numPart = nPart;
        S.numVox  = nVox;
        trueRDM=repmat(rsa_vectorizeRDM(D),2,1);
        trueDist = calcDist(trueRDM);
        M=makeModel('sameRDM',D,nCond); 
        for s=noise
            [data(1),partVec,condVec] = pcm_generateData(M,0.1,S,1,1,0);
            [data(2),partVec,condVec] = pcm_generateData(M,0.1,S,1,1,s);
            N.calcDist = calcDist(trueRDM);
            calcRDM = makeRDM_crossval(data,partVec,condVec);
            N.calcDist = calcDist(calcRDM);
            % dist type labelling
            N.distType        = [1:4]';
            N.distLabel(1,:)  = {'correlation'};
            N.distLabel(2,:)  = {'cosine'};
            N.distLabel(3,:)  = {'euclidean'};
            N.distLabel(4,:)  = {'distCorr'};
            N.noise1 = repmat(1,4,1);
            N.noise2 = repmat(s,4,1);
            NN = addstruct(NN,N);
        end

        figure
        plt.line(NN.noise2,NN.calcDist,'split',NN.distType,'leg',{'correlation','cosine','euclidean','distCorr'},'leglocation','northwest');
        ylabel('distance between identical RDMs');
        xlabel('noise proportion of RDM 1 vs. 2');
    
    case 'singleReg'
        nCond = 5;      
        snr = [0.1,0.5,1,5,10];
        nPart = 9;
        nVox = 1000;
        vararginoptions(varargin,{'nCond','signal','snr'});
 
        v = abs(randn(1,nCond*(nCond-1)/2));
        D = rsa_squareRDM(v);
        H = eye(nCond) - 1/nCond;
        Gc = -0.5*H*D*H';
        trueRDM = rsa_vectorizeRDM(D);
        trueG   = rsa_vectorizeIPMfull(Gc);
        NN=[]; 
        for s=snr
            for n=1:100
                [Y1,partVec,condVec]    = makePatterns('G',Gc,'nVox',nVox,'nPart',nPart,'noise',s);
                Ge1                     = pcm_estGCrossval(Y1,partVec,condVec);
                N.G_calc                = rsa_vectorizeIPMfull(Ge1);
                N.Dist_calc(1,:)        = rsa.distanceLDC(Y1,partVec,condVec);
                N.G_true                = trueG;
                N.Dist_true             = trueRDM;
                N.Dist_T                = mean(N.Dist_true,2);
                N.Dist_C                = mean(N.Dist_calc,2);
                N.snr                   = s;
                N.numSim                = n;
                NN = addstruct(NN,N);
            end
        end

        figure
        plt.line(NN.snr,NN.Dist_C);
        hold on;
        drawline(NN.Dist_T(1),'dir','horz');
        ylabel('mean distance');
        xlabel('noise');
        figure
        subplot(261)
        imagesc(rsa_squareIPMfull(N.G_true(1,:)));
        title('true G');
        for i=1:5
            subplot(2,6,i+1)
            imagesc(rsa_squareIPMfull(mean(NN.G_calc(NN.snr==snr(i),:))));
            title(sprintf('G - snr %2.1f', snr(i)));
        end
        subplot(267)
        imagesc(rsa_squareRDM(N.Dist_true(1,:)));
        title('true distance');
        for i=1:5
            subplot(2,6,7+i)
            imagesc(rsa_squareRDM(mean(NN.Dist_calc(NN.snr==snr(i),:))));
            title(sprintf('dist - snr %2.1f', snr(i)));
        end    
    case 'twoReg'
        nCond = 5;      
        snr = [0.001,0.005,0.01:0.01:0.1,0.2,0.5,1];
        nPart = 9;
        nVox = 1000;
        RDMsim = 1; % 1 - same, 0.5 - different; 0 - one has zeros only
        vararginoptions(varargin,{'nCond','signal','snr','RDMsim'});
 
        v = abs(randn(1,nCond*(nCond-1)/2));
        D = rsa_squareRDM(v);
        H = eye(nCond) - 1/nCond;
        Gc1 = -0.5*H*D*H';
        trueRDM(1,:)=rsa_vectorizeRDM(D);
        switch RDMsim
            case 1
                Gc2=Gc1;
                trueRDM(2,:) = trueRDM(1,:);
            case 0
                Gc2=zeros(5,5);
                trueRDM(2,:) = zeros(size(trueRDM(1,:)));
            case 0.5
                v = abs(randn(1,nCond*(nCond-1)/2));
                D = rsa_squareRDM(v);
                H = eye(nCond) - 1/nCond;
                Gc2 = -0.5*H*D*H';
                trueRDM(2,:)=rsa_vectorizeRDM(D);
        end
        trueRegDist = calcDist(trueRDM);

        NN=[]; 
        for s=snr
            for n=1:100
                [data{1},partVec,condVec]    = makePatterns('G',Gc1,'nVox',nVox,'nPart',nPart,'snr',s);
                [data{2},partVec,condVec]    = makePatterns('G',Gc2,'nVox',nVox,'nPart',nPart,'snr',0.1);
                
                N.trueRegDist           = trueRegDist;
                calcRDM                 = makeRDM_crossval(data,partVec,condVec);
                N.calcRegDist           = calcDist(calcRDM);
                % dist type labelling
                N.distType        = [1:4]';
                N.distLabel(1,:)  = {'correlation'};
                N.distLabel(2,:)  = {'cosine'};
                N.distLabel(3,:)  = {'euclidean'};
                N.distLabel(4,:)  = {'distCorr'};
                N.snr1            = repmat(s,4,1);
                N.snr2            = repmat(10,4,1);
                N.numSim          = repmat(n,4,1);
                NN = addstruct(NN,N);
            end
        end
        figure
        subplot(211)
        plt.line(NN.snr1,NN.calcRegDist,'split',NN.distType,'leg',distLabels,'leglocation','northeast');
        xlabel('snr of region with variable snr');
        ylabel('estimated distance');
        title(sprintf('two regions one varies snr - type%2.1d of RDMs',RDMsim));   
        subplot(212)
        plt.line(NN.snr1,NN.trueRegDist-NN.calcRegDist,'split',NN.distType,'leg',distLabels,'leglocation','northeast');
        xlabel('snr of region with variable snr');
        ylabel('true-estimated distance');
    case 'twoReg_sharedNoise'
        nCond = 5;      
        numSim = 100;
      %  noise_s = [0:0.01:0.09,0.1:0.1:0.8];
        noise_s = [0:1,5,10:10:50];
        nPart   = 8;
        nVox    = 1000;
        RDMtype = 1; % 1 - same, 0.5 - different; 0 - one has zeros only
        vararginoptions(varargin,{'nCond','signal','noise_s','RDMtype','numSim'});
        
        switch RDMtype
            case 1 % G1 = G2 = 0
                Gc1=zeros(5);
                Gc2=zeros(5);
                D1=zeros(5);
                D2=zeros(5); 
            case 2 % G1 = G2
                v1 = abs(randn(1,nCond*(nCond-1)/2));
                D1 = rsa_squareRDM(v1);
                D2 = D1;
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2=Gc1;
            case 3 % G1 orthogonal G2
                D=zeros(nCond); D1=D; D2=D;
                D1(1:2,1:5)=1;
                D1(3:5,1:2)=1;
                D2(3:5,3:5)=1;
                D1(1:nCond+1:end)=0;
                D2(1:nCond+1:end)=0;
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2 = -0.5*H*D2*H';
            case 4 % G1 ~= G2
                v1 = abs(randn(1,nCond*(nCond-1)/2));
                D1 = rsa_squareRDM(v1);
                v2 = abs(randn(1,nCond*(nCond-1)/2));
                D2 = rsa_squareRDM(v2);
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2 = -0.5*H*D2*H';
        end
        % calculate true RDM
        trueRDM(1,:)=rsa_vectorizeRDM(D1);
        trueRDM(2,:)=rsa_vectorizeRDM(D2);
        trueRegDist = calcDist(trueRDM); % true reg distance
        t = trueRDM';
        tRDM = t(:); % vectorize across two RDMs
        % make models for data generation
        M{1}=makeModel('sameRDM',Gc1,nCond);
        M{2}=makeModel('sameRDM',Gc2,nCond);
        S.numPart = nPart;
        S.numVox  = nVox;
        NN=[]; RR=[]; DD=[];
        for s=noise_s
            for n=1:numSim
                [data(1),partVec,condVec] = pcm_generateData(M{1},M{1}.theta,S,1,1,0.1); %signal 10, noise 0
                [data(2),partVec,condVec] = pcm_generateData(M{2},M{2}.theta,S,1,1,0.1);
               % [data{1},partVec,condVec] = makePatterns('G',M{1}.Ac,'nPart',nPart,'nVox',nVox,'signal',1,'noise',0); %signal 10, noise 0
               % [data{2},partVec,condVec] = makePatterns('G',M{2}.Ac,'nPart',nPart,'nVox',nVox,'signal',1,'noise',0);
                % add shared noise across regions
                Z = normrnd(1,0.2,size(data{1},1),2*nVox);
                g = 1; % scalar
                P = zeros(2*nVox)+0.5; % voxel covariance matrix
                P(1:2*nVox+1:end)=ones(2*nVox,1); % 1 on diag
                Zn = sqrt(g)*Z*sqrtm(P);     % shared noise matrix across reg
                data{1} = data{1} + s*Zn(:,1:nVox);
                data{2} = data{2} + s*Zn(:,nVox+1:2*nVox);
                % predict the multivariate dependency data{1}->data{2}
                [R2_all,r_all]      = multiDepend(data{1},data{2},partVec,condVec,'type','all');
                [R2_redA,r_redA]    = multiDepend(data{1},data{2},partVec,condVec,'type','reduceA');
                [R2_redAB,r_redAB]  = multiDepend(data{1},data{2},partVec,condVec,'type','reduceAB');
                N.R2_all    = repmat(R2_all,4,1);
                N.r_all     = repmat(r_all,4,1);
                N.R2_redA   = repmat(R2_redA,4,1);
                N.r_redA    = repmat(r_redA,4,1);
                N.R2_redAB  = repmat(R2_redAB,4,1);
                N.r_redAB   = repmat(r_redAB,4,1);
                
                % distance metrics
                N.trueRegDist     = trueRegDist;
                calcRDM           = makeRDM_crossval(data,partVec,condVec);
                N.calcRegDist     = calcDist(calcRDM);  
                % dist type labelling
                N.distType        = [1:4]';
                N.distLabel(1,:)  = {'correlation'};
                N.distLabel(2,:)  = {'cosine'};
                N.distLabel(3,:)  = {'euclidean'};
                N.distLabel(4,:)  = {'distCorr'};
                N.sharedNoise     = repmat(s,4,1);
                N.regCorr         = repmat(corr(data{1}(:),data{2}(:)),4,1);
                N.numSim          = repmat(n,4,1);
                NN = addstruct(NN,N);   
                % RDMs
                c = calcRDM';
                cRDM = c(:);
                R.calcRDM       = cRDM';
                R.trueRDM       = tRDM';
                R.sharedNoise = s;
                RR = addstruct(RR,R);
            end
            fprintf('Done %d/%d: \tsimulations for noise level %d.\n',find(s==noise_s),length(noise_s),s);
        end
        save(fullfile(baseDir,sprintf('sim_sharedNoise_RDM_type%d',RDMtype)),'-struct','NN');
        save(fullfile(baseDir,sprintf('sim_sharedNoise_dist_RDM_type%d',RDMtype)),'-struct','RR');
    case 'PLOT_sharedNoise_dist'
        RDMtype = 1;
        vararginoptions(varargin,{'RDMtype'});
        N = load(fullfile(baseDir,sprintf('sim_sharedNoise_RDM_type%d',RDMtype)));
        R = load(fullfile(baseDir,sprintf('sim_sharedNoise_dist_RDM_type%d',RDMtype)));
        figure
        subplot(441)
        imagesc(rsa_squareRDM(R.trueRDM(1,[1:10]))); colorbar;
        title('true RDM reg1');
        subplot(442)
        imagesc(rsa_squareRDM(R.trueRDM(2,[11:20]))); colorbar;
        title('true RDM reg2');
        subplot(443)
        plt.scatter(N.sharedNoise,N.calcRegDist,'split',N.distType,'leg',distLabels,'style',sCol,'leglocation','northeast');
        xlabel('shared noise across regions');
        ylabel('estimated distance');
        subplot(444)
        if RDMtype == 1
            N.trueRegDist = zeros(size(N.trueRegDist));
        end
        if RDMtype <3
            plt.scatter(N.sharedNoise,N.calcRegDist-N.trueRegDist,'split',N.distType,'leg',distLabels,'style',sCol,'leglocation','northeast');
            hold on;
            drawline(0,'dir','horz');
            ylabel('corrected distance -');
        else
            plt.scatter(N.sharedNoise,N.calcRegDist./N.trueRegDist,'split',N.distType,'leg',distLabels,'style',sCol,'leglocation','northeast');
            hold on;
            drawline(1,'dir','horz');
            ylabel('corrected distance /');
        end
        
        t=N.trueRegDist(N.sharedNoise==0);
        subplot(445)
        plt.scatter(N.sharedNoise,N.calcRegDist,'split',N.distType,'subset',N.distType==1,'style',style.custom({blue}),'leg',distLabels(1));
        hold on;
        drawline(t(1),'dir','horz');
        title(sprintf('true dist %2.1f',t(1)));
        subplot(446)
        plt.scatter(N.sharedNoise,N.calcRegDist,'split',N.distType,'subset',N.distType==2,'style',style.custom({mediumblue}),'leg',distLabels(2));
        hold on;
        drawline(t(2),'dir','horz');
        title(sprintf('true dist %2.1f',t(2)));
        subplot(447)
        plt.scatter(N.sharedNoise,N.calcRegDist,'split',N.distType,'subset',N.distType==3,'style',style.custom({mediumred}),'leg',distLabels(3));
        hold on;
        drawline(t(3),'dir','horz');
        title(sprintf('true dist %2.1f',t(3)));
        subplot(448)
        plt.scatter(N.sharedNoise,N.calcRegDist,'split',N.distType,'subset',N.distType==2,'style',style.custom({lightblue}),'leg',distLabels(4));
        hold on;
        drawline(t(4),'dir','horz');
        title(sprintf('true dist %2.1f',t(4)));
        subplot(449)
        T = getrow(R,R.sharedNoise==0);
        imagesc(rsa_squareRDM(mean(T.calcRDM(:,[1:10])))); colorbar;
        title('calc RDM1 - noise 0');
        subplot(4,4,10)
        imagesc(rsa_squareRDM(mean(T.calcRDM(:,[11:20])))); colorbar;
        title('calc RDM2 - noise 0');
        subplot(4,4,13)
        T = getrow(R,R.sharedNoise==50);
        imagesc(rsa_squareRDM(mean(T.calcRDM(:,[1:10])))); colorbar;
        title('calc RDM1 - noise 50');
        subplot(4,4,14)
        imagesc(rsa_squareRDM(mean(T.calcRDM(:,[11:20])))); colorbar;
        title('calc RDM2 - noise 50');
    case 'PLOT_sharedNoise_cov_v1'
        RDMtype = 1;
        vararginoptions(varargin,{'RDMtype'});
        R = load(fullfile(baseDir,sprintf('sim_sharedNoise_dist_RDM_type%d',RDMtype)));
        figure
        sn = [0,10,30,50];
        idx=1;
        for i=1:length(sn)
            T = getrow(R,R.sharedNoise==sn(i));
            t = nanmean(T.calcRDM,1);
            subplot(length(sn),4,idx)
            imagesc(t(1:10)'*t(1:10)); colorbar;
            title(sprintf('d1^T * d1 - %d noise',sn(i)));
            subplot(length(sn),4,idx+1)
            imagesc(t(11:20)'*t(11:20)); colorbar;
            title('d2^T * d2');
            subplot(length(sn),4,idx+2)
            imagesc(t(1:10)'*t(11:20)); colorbar;
            title('d1^T * d2');
            subplot(length(sn),4,idx+3)
            imagesc((t(1:10)'*t(11:20))/ssqrt((t(1:10)'*t(1:10))*t(11:20)'*t(11:20))); colorbar;
            title('corr d1 d2');
            idx=idx+4;
        end
    case 'PLOT_sharedNoise_cov_v2'
        RDMtype = 1;
        vararginoptions(varargin,{'RDMtype'});
        R = load(fullfile(baseDir,sprintf('sim_sharedNoise_dist_RDM_type%d',RDMtype)));
        figure
        sn = [0,10,30,50];
        idx=1;
        for i=1:length(sn)
            T = getrow(R,R.sharedNoise==sn(i));
            for j=1:size(T.calcRDM,1)
                d1(j,:)=T.calcRDM(j,[1:10])*T.calcRDM(j,[1:10])';
                d2(j,:)=T.calcRDM(j,[11:20])*T.calcRDM(j,[11:20])';
                d12(j,:)=T.calcRDM(j,[1:10])*T.calcRDM(j,[11:20])';
                dc(j,:)=(d12(j)/ssqrt(d1(j)*d2(j)));
            end
            if any(isnan(dc))
                dc(isnan(dc))=0;
            end
            subplot(length(sn),4,idx);
            plt.scatter(d1,d12);
            xlabel('var(d1)');
            ylabel('cov(d1,d2)');
            title(sprintf('noise %d',sn(i)));
            subplot(length(sn),4,idx+1);
            plt.scatter(d2,d12);
            xlabel('var(d2)');
            ylabel('cov(d1,d2)');
            subplot(length(sn),4,idx+2);
            plt.scatter(d1,dc);
            xlabel('var(d1)');
            ylabel('corr(d1,d2)');
            subplot(length(sn),4,idx+3);
            plt.scatter(d12,dc);
            xlabel('cov(d1,d2)');
            ylabel('corr(d1,d2)');
            idx=idx+4;
        end
    case 'PLOT_sharedNoise_hist'
        RDMtype = 1;
        vararginoptions(varargin,{'RDMtype'});
        R = load(fullfile(baseDir,sprintf('sim_sharedNoise_dist_RDM_type%d',RDMtype)));
        sn = [0,5,20,30,50];
        T = getrow(R,ismember(R.sharedNoise,sn));
        for j=1:size(T.calcRDM,1)
            d1(j,:)=T.calcRDM(j,[1:10])*T.calcRDM(j,[1:10])';
            d2(j,:)=T.calcRDM(j,[11:20])*T.calcRDM(j,[11:20])';
            d12(j,:)=T.calcRDM(j,[1:10])*T.calcRDM(j,[11:20])';
            dc(j,:)=(d12(j)/ssqrt(d1(j)*d2(j)));
        end
        if any(isnan(dc))
            dc(isnan(dc))=0;
        end
        figure
        subplot(2,2,1);
        plt.hist(d1,'split',T.sharedNoise);
        title('var(d1)');
        subplot(2,2,2);
        plt.hist(d2,'split',T.sharedNoise);
        title('var(d2)');
        subplot(2,2,3);
        plt.hist(d12,'split',T.sharedNoise);
        title('cov(d1,d2)');
        subplot(2,2,4);
        plt.hist(dc,'split',T.sharedNoise);
        title('corr(d1,d22)');
    case 'PLOT_sharedNoise_r'
        RDMtype=1;
        vararginoptions(varargin,{'RDMtype'});
        N = load(fullfile(baseDir,sprintf('sim_sharedNoise_RDM_type%d',RDMtype)));
        
        N = getrow(N,N.distType==1);
        figure
        subplot(231)
        plt.scatter(N.sharedNoise,N.R2_all);
        title('R2-all');
        subplot(232)
        plt.scatter(N.sharedNoise,N.R2_redA);
        title('R2-redA');
        subplot(233)
        plt.scatter(N.sharedNoise,N.R2_redAB);
        title('R2-redAB');
        subplot(234)
        plt.scatter(N.sharedNoise,N.r_all);
        title('r-all');
        subplot(235)
        plt.scatter(N.sharedNoise,N.r_redA);
        title('r-redA');
        subplot(236)
        plt.scatter(N.sharedNoise,N.r_redAB);
        title('r-redAB');
        
            
    case 'twoReg_regress'
        nCond = 5;      
        numSim = 100;
        noise_s = [0:1,5,10:10:50];
        nPart   = 8;
        nVox    = 1000;
        RDMtype = 1; % 1 - same, 0.5 - different; 0 - one has zeros only
        vararginoptions(varargin,{'nCond','signal','noise_s','RDMtype','numSim'});
        
        switch RDMtype
            case 1 % G1 = G2 = 0
                Gc1=zeros(5);
                Gc2=zeros(5);
                D1=zeros(5);
                D2=zeros(5); 
            case 2 % G1 = G2
                v1 = abs(randn(1,nCond*(nCond-1)/2));
                D1 = rsa_squareRDM(v1);
                D2 = D1;
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2=Gc1;
            case 3 % G1 orthogonal G2
                D=zeros(nCond); D1=D; D2=D;
                D1(1:2,1:5)=1;
                D1(3:5,1:2)=1;
                D2(3:5,3:5)=1;
                D1(1:nCond+1:end)=0;
                D2(1:nCond+1:end)=0;
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2 = -0.5*H*D2*H';
            case 4 % G1 ~= G2
                v1 = abs(randn(1,nCond*(nCond-1)/2));
                D1 = rsa_squareRDM(v1);
                v2 = abs(randn(1,nCond*(nCond-1)/2));
                D2 = rsa_squareRDM(v2);
                H = eye(nCond) - 1/nCond;
                Gc1 = -0.5*H*D1*H';
                Gc2 = -0.5*H*D2*H';
        end
        % calculate true RDM
        trueRDM(1,:)=rsa_vectorizeRDM(D1);
        trueRDM(2,:)=rsa_vectorizeRDM(D2);
        trueRegDist = calcDist(trueRDM); % true reg distance
        t = trueRDM';
        tRDM = t(:); % vectorize across two RDMs
        % make models for data generation
        M{1}=makeModel('sameRDM',Gc1,nCond);
        M{2}=makeModel('sameRDM',Gc2,nCond);
        S.numPart = nPart;
        S.numVox  = nVox;
        NN=[]; RR=[];
        for s=noise_s
            for n=1:numSim
                [data(1),partVec,condVec] = pcm_generateData(M{1},M{1}.theta,S,1,1,0); %signal 1, noise 0
                [data(2),partVec,condVec] = pcm_generateData(M{2},M{2}.theta,S,1,1,0);
                % add shared noise across regions
                Z = normrnd(1,0.2,size(data{1},1),2*nVox);
                g = 1; % scalar
                P = zeros(2*nVox)+0.5; % voxel covariance matrix
                P(1:2*nVox+1:end)=ones(2*nVox,1); % 1 on diag
                Zn = sqrt(g)*Z*sqrtm(P);     % shared noise matrix across reg
                data{1} = data{1} + s*Zn(:,1:nVox);
                data{2} = data{2} + s*Zn(:,nVox+1:2*nVox);
                % predict the multivariate dependency data{1}->data{2}
                [N.R2_all,N.r_all]      = multiDepend(data{1},data{2},partVec,condVec,'type','all');
                [N.R2_redA,N.r_redA]    = multiDepend(data{1},data{2},partVec,condVec,'type','reduceA');
                [N.R2_redAB,N.r_redAB]  = multiDepend(data{1},data{2},partVec,condVec,'type','reduceAB');
                N.trueRegDist     = trueRegDist;
                calcRDM           = makeRDM_crossval(data,partVec,condVec);
                N.calcRegDist     = calcDist(calcRDM);  
                % dist type labelling
                N.distType        = [1:4]';
                N.distLabel(1,:)  = {'correlation'};
                N.distLabel(2,:)  = {'cosine'};
                N.distLabel(3,:)  = {'euclidean'};
                N.distLabel(4,:)  = {'distCorr'};
                N.sharedNoise     = repmat(s,4,1);
                N.regCorr         = repmat(corr(data{1}(:),data{2}(:)),4,1);
                N.numSim          = repmat(n,4,1);
                NN = addstruct(NN,N);   
                % RDMs
                c = calcRDM';
                cRDM = c(:);
                R.calcRDM       = cRDM';
                R.trueRDM       = tRDM';
                R.sharedNoise = s;
                RR = addstruct(RR,R);
            end
            fprintf('Done %d/%d: \tsimulations for noise level %d.\n',find(s==noise_s),length(noise_s),s);
        end
        save(fullfile(baseDir,sprintf('sim_sharedNoise_regress_RDM_type%d',RDMtype)),'-struct','NN');
        save(fullfile(baseDir,sprintf('sim_sharedNoise_regress_dist_RDM_type%d',RDMtype)),'-struct','RR');
    case 'regress-dist'
        nCond = 5;
        noise_s = [0:1:10];
        nPart = 8;
        nVox = 1000;
        RDMsim = 'diff'; % 1 - same, 0.5 - different; 0 - one has zeros only
        numDist = 15;
        vararginoptions(varargin,{'nCond','signal','noise_s','RDMsim','ms'});
        
        
        for t=1:numDist
            v1 = abs(randn(1,nCond*(nCond-1)/2));
            D1 = rsa_squareRDM(v1);
            H = eye(nCond) - 1/nCond;
            Gc1 = -0.5*H*D1*H';
            trueRDM(1,:) = rsa_vectorizeRDM(D1);
            switch RDMsim
                case 'same'
                    Gc2=Gc1;
                    trueRDM(2,:) = trueRDM(1,:);
                case 'zero'
                    Gc2=zeros(5,5);
                    trueRDM(2,:) = zeros(size(trueRDM(1,:)));
                case 'diff'
                    v = abs(randn(1,nCond*(nCond-1)/2));
                    D = rsa_squareRDM(v);
                    H = eye(nCond) - 1/nCond;
                    Gc2 = -0.5*H*D*H';
                    trueRDM(2,:)=rsa_vectorizeRDM(D);
            end
            trueRegDist = calcDist(trueRDM);
            t = trueRDM';
            tRDM = t(:);
            % make models for data generation
            M{1}=makeModel('sameRDM',Gc1,nCond);
            M{2}=makeModel('sameRDM',Gc2,nCond);
            S.numPart = nPart;
            S.numVox  = nVox;
            NN=[]; RR=[];
            for s=noise_s
                for n=1:25
                    [data(1),partVec,condVec] = pcm_generateData(M{1},M{1}.theta,S,1,1,0); %signal 10, noise 0
                    [data(2),partVec,condVec] = pcm_generateData(M{2},M{2}.theta,S,1,1,0);
                    % add shared noise across regions
                    Z = normrnd(1,0.2,size(data{1},1),2*nVox);
                    g = 1; % scalar
                    P = zeros(2*nVox)+0.5; % voxel covariance matrix
                    P(1:2*nVox+1:end)=ones(2*nVox,1); % 1 on diag
                    Zn = sqrt(g)*Z*sqrtm(P);     % shared noise matrix across reg
                    data{1} = data{1} + s*Zn(:,1:nVox);
                    data{2} = data{2} + s*Zn(:,nVox+1:2*nVox);
                    
                    % split data into partitions
                    part{1} = data{1}(rem(partVec,2)==1,:);
                    part{2} = data{1}(rem(partVec,2)==0,:);
                    part{3} = data{2}(rem(partVec,2)==1,:);
                    part{4} = data{2}(rem(partVec,2)==1,:);
                    pVec = partVec(1:end/2);
                    cVec = condVec(1:end/2);
                    % split RDM estimates
                    partRDM = makeRDM_crossval(part,pVec,cVec);
                    X_train = partRDM(1,:)';
                    Y_train = partRDM(3,:)';
                    X_test  = partRDM(2,:)';
                    Y_test  = partRDM(4,:)';
                    % estimate R2,r
                    beta = pinv(X_train)*Y_train;
                    res = Y_test - (X_test.*beta);
                    SSR = sum(res.^2);
                    SST = sum(Y_test.^2);
                    R2 = 1 - SSR/SST;
                    R.R2 = R2;
                    R.r  = corr(X_test.*beta,Y_train);
                    R.beta          = beta;
                    % estimate dist
                    estRDM          = makeRDM_crossval(data,partVec,condVec);
                    calcD           = calcDist(estRDM);
                    R.estEucD       = calcD(3);
                    R.estCorrD      = calcD(1);
                    R.trueEucD      = trueRegDist(3);
                    R.trueCorrD     = trueRegDist(1);
                    R.sharedNoise   = s;
                    R.trueRDM       = tRDM';
                    calcRDM         = makeRDM_crossval(data,partVec,condVec);
                    c               = calcRDM';
                    cRDM            = c(:);
                    R.estRDM       = cRDM';
                    trainRDM        = partRDM([1,3],:)';
                    testRDM         = partRDM([2,4],:)';
                    trR             = trainRDM(:);
                    teR             = testRDM(:);
                    R.trainRDM      = trR(:)';
                    R.testRDM       = teR(:)';
                    RR = addstruct(RR,R);
                end
                fprintf('Done %d/%d: \tsimulations for noise level %d.\n',find(s==noise_s),length(noise_s),s);
            end
        end
        save(fullfile(baseDir,'regress_dist'),'-struct','RR');
        keyboard;
    case 'plot_sharedNoise'
        RDMsim=0.5;
        T=load(fullfile(baseDir,sprintf('simulations_sharedNoise_RDMtype_0.5.mat')));
        R=load(fullfile(baseDir,sprintf('simulations_sharedNoise_dist_RDMtype_0.5.mat')));
        noise=unique(T.sharedNoise);
        figure
        idx=1;
        for n=noise'
            D = getrow(R,R.sharedNoise==n);
            d = mean(D.calcRDM);
            c = d'*d;
            subplot(numel(unique(noise)),3,idx); 
            imagesc(c(1:10,1:10));
            colorbar;
            title(sprintf('var1 noise %d',n));
            subplot(numel(unique(noise)),3,idx+1); 
            imagesc(c(11:20,11:20));
            colorbar;
            title('var2');
            subplot(numel(unique(noise)),3,idx+2);
            imagesc(c(1:10,11:20));
            colorbar;
            title('cov');
            p=rsa_vectorizeIPM(c(1:10,1:10));
            q=rsa_vectorizeIPM(c(11:20,11:20));
            c_pq = corr(p',q');
            fprintf('noise %d correlation: %2.1f\n',n,c_pq);
          %  subplot(numel(unique(noise)),3,idx+3)
          %  sl=plt.scatter(rsa_vectorizeIPMfull(c(1:10,1:10))',rsa_vectorizeIPMfull(c(1:10,11:20))');
          %  xlabel('var1'); ylabel('cov'); title(sprintf('slope %2.1f',sl));
          %  subplot(numel(unique(noise)),5,idx+4)
          %  sl=plt.scatter(rsa_vectorizeIPMfull(c(11:20,11:20))',rsa_vectorizeIPMfull(c(1:10,11:20))');
          %  xlabel('var2'); ylabel('cov'); title(sprintf('slope %2.1f',sl));
            idx=idx+3;
        end
        keyboard;
    case 'normalize'
        nCond = 5;     
        D1=randn(nCond);
        D2=randn(nCond);
        D1(1:nCond+1:end)=zeros(nCond,1);
        D2(1:nCond+1:end)=zeros(nCond,1);
        
        % normalize
        s1 = mean(mean(D1));
        s2 = mean(mean(D2));

        %
        % D2 = k*D1+res
        % D2*s2 = D1*s1 + res
        % D2 = (s1/s2)*D1+res;
        % k = s1/s2
        % res = D2-k*D1; 
        k = s1/s2;
        res = D2-k*D1;
        
        [U,S,V]=svd(res);
        % U*V' - rotation
        % S - non-isomorphic scaling
        
       % D2==k*D1+U*S*V'; (up to numeric precision)
        
        D1_meanSub  = bsxfun(@minus,D1,mean(D1,2));
        D2_meanSub  = bsxfun(@minus,D2,mean(D2,2));
        D1_standar = bsxfun(@times,D1,1./std(D1,[],2));
        
        
    case 'PLOT_dist_hist'
        simuType='sameRDM'; % sameRDM or randomRDM, combRDM
        noiseLevels=[0,0.5,1];
        noiseType=1; % 1 or 0; 1 - same noise type, 0 opposite
        vararginoptions(varargin,{'simuType','simuType','noiseLevels','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s.mat',simuType)));
          for s=1:size(averageType,2);
           D = getrow(DD,DD.averageType==s); 
            figure
            for n=1:size(noiseLevels,2)
                if noiseType==1
                    d=getrow(D,D.noise1==noiseLevels(n) & D.noise2==noiseLevels(n));
                else
                    d=getrow(D,D.noise1==noiseLevels(n) & D.noise1+D.noise2==1);    
                end  
                subplot(1,size(noiseLevels,2),n)
                tLabel = sprintf('Corr noise-%2.1f %s',noiseLevels(n),averageType{s});
                hist_corr(d.calcDist,d.distType,'legLabels',distLabels,'titleLabel',tLabel);          
            end
          end
    case 'PLOT_calc_trueDist'
        simuType='sameRDM'; % sameRDM, randomRDM or combRDM
        noiseType=1; % 1=same or 0=opposite
        vararginoptions(varargin,{'simuType','rdmType','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
        figure
        for s=1:size(averageType,2);
            D = getrow(DD,DD.averageType==s);
            if noiseType==1
                D = getrow(D,D.noise1==D.noise2);
            else
                D = getrow(D,D.noise1+D.noise2==1);
            end
            if strcmp(simuType,'combRDM')
                D.calcDist = D.calcDist(:,2);
                D.trueDist = D.trueDist(:,2);
            end
            subplot(1,size(averageType,2),s)
            plt.scatter(D.noise1,D.calcDist-D.trueDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title(sprintf('Corr %s %s',averageType{s},simuType));
            drawline(0,'dir','horz');
            xlabel('noise levels'); ylabel('estimated - true distance');
            plt.match('y');
        end 
    case 'PLOT_evaluate'
        simuType = 'randomRDM';
        noiseType=1; % 1=same or 0=opposite
        vararginoptions(varargin,{'simuType','rdmType','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('evaluate_%s',simuType)));
        for s=1:size(averageType,2);
            figure
            D = getrow(DD,DD.averageType==s);
            if noiseType==1
                D = getrow(D,D.noise1==D.noise2);
            else
                D = getrow(D,D.noise1+D.noise2==1);
            end
            subplot(151)
            plt.scatter(D.noise1,D.trueDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title('true distance');
            xlabel('noise levels');
            subplot(152)
            plt.scatter(D.noise1,D.calcDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title(sprintf('calculated distance - %s type',averageType{s}));
            subplot(153)
            plt.scatter(D.noise1,D.bias,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title('bias');
            drawline(0,'dir','horz','linestyle','--');
            subplot(154)
            plt.scatter(D.noise1,D.var,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            drawline(0,'dir','horz','linestyle','--');
            title('variance');
            subplot(155)
            plt.scatter(D.noise1,D.rmse,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            drawline(0,'dir','horz','linestyle','--');
            title('rmse');
        end
        
    case 'PLOT_combRDM'
        noiseLevels=[0,0.5,1];
        rdmType={'allPart','averagePart','crossval'};
        vararginoptions(varargin,{'noiseLevels','rdmType'});
        
        
        DD = load(fullfile(baseDir,'simulations_combRDM'));
        for n=1:length(noiseLevels)
            D = getrow(DD,DD.noise==noiseLevels(n));
            for s=1:2
                if s==1
                    xvar = D.weight1;
                    yvar = D.corr(:,2);
                else
                    xvar = D.weight2;
                    yvar = D.corr(:,3);
                end
                figure
                for r=1:length(rdmType)
                    subplot(1,length(rdmType),r)
                    plt.scatter(xvar,yvar,'split',D.corrType,'subset',D.rdmType==r,'leg',{'rsa-intercept','rsa-noIntercept','dist'},'leglocation','northeast');
                    xlabel('weight for RDM creation');
                    ylabel('corr to RDM with weights on x');
                    title(sprintf('Corr noise-%1.1f %s',noiseLevels(n),rdmType{r}));
                end
                plt.match('y');
            end
        end
    case 'empiricalExample'
       D=load(fullfile(baseDir,'RDMs'));
       regLabels={'ba1','ba2','ba3','ba4','ba6'};
       
       % extract relevant field
       for i=1:length(regLabels)
           % create RDM
           RDM{i}=rsa_squareRDM(getfield(D,regLabels{i}));
       end
       
       corrRSA  = rsa_calcCorrRDMs(RDM);
       corrDist = rsa_calcDistCorrRDMs(RDM);
       
       acrReg_rsa  = rsa_squareRDM(corrRSA);
       acrReg_dist = rsa_squareRDM(corrDist);
       
       figure;
       subplot(1,2,1);
       imagesc(acrReg_rsa,[0 1.5]);
       title('Across regions RSA correlation');
       subplot(1,2,2);
       imagesc(acrReg_dist,[0 1.5]);
       title('Across regions DIST correlation');
     
    case 'run_job'
      rsa_connect('twoReg_sharedNoise','RDMtype',1);
      fprintf('Done 2\n');
      rsa_connect('twoReg_sharedNoise','RDMtype',2);
      fprintf('Done 2\n');
      rsa_connect('twoReg_sharedNoise','RDMtype',3);
      fprintf('Done 3\n');
      rsa_connect('twoReg_sharedNoise','RDMtype',4);
      fprintf('Done 4\n');
      
      
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions

function M = makeModel(rdmType,D,nCond)
M.type       = 'feature';
%M.numGparams = 1;
M.numGparams = 1;
M.theta      = 1; 
switch rdmType
    case 'sameRDM'
        M.Ac=D;
    case 'randomRDM'
        M.Ac=rand(nCond);
    case 'combRDM'
        M.Ac=D;
end
end

function dist = calcDist(rdm)
% calculate distance metric from the input
% N datasets, D distances
% input: N x D matrix
% dist types:
% 1) correlation
% 2) cosine
% 3) euclidean
% 4) distance correlation

dist(1,:)  = pdist(rdm,'correlation');
dist(2,:)  = pdist(rdm,'cosine');
dist(3,:)  = pdist(rdm,'euclidean');
dist(4,:) = rsa_calcDistCorrRDMs(rdm);

end

function rdm   = makeRDM_average(data,condVec)
% function rdm = makeRDM_average(data,condVec)
% makes RDM matrix from euclidean distances (not crossvalidated)
    % make indicator matrix for estimating mean pattern
    X = indicatorMatrix('identity_p',condVec);
    for st=1:size(data,2)
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm(st,:)=pdist(D{st});
    end; 
end
function rdm   = makeRDM_allPart(data)
% function rdm = makeRDM_allPart(data)
% makes RDM matrix from euclidean distances across all runs (not averaging)
    for st=1:size(data,2)
        % make RDM from Euclidean distances
        rdm(st,:)=pdist(data{st});
    end;
end
function rdm   = makeRDM_crossval(data,partVec,condVec)
% function rdm   = makeRDM_crossval(data,partVec,condVec)
% makes crossvalidated RDM matrix
numDataset=size(data,2);
for st=1:numDataset
    % calculate crossvalidated squared Euclidean distances
    rdm(st,:)=rsa.distanceLDC(data{st},partVec,condVec);
end;

end

function hist_corr(corr,corrType,varargin)
% function hist_corr(corr_rsa,corr_dist,vararginoptions)
% plots the distribution of rsa and distance correlations
% varargin - optional level of noise for the title
vararginoptions(varargin,{'legLabels','titleLabel'});

    plt.hist(corr,'split',corrType,'leg',legLabels,'leglocation','northeast');
    drawline(mean(corr(corrType==1)),'dir','vert','color','b');
    drawline(mean(corr(corrType==2)),'dir','vert','color','g');
    drawline(mean(corr(corrType==3)),'dir','vert','color','r');
    drawline(mean(corr(corrType==4)),'dir','vert','color','y');
    xlabel('Estimated distance');
    title(titleLabel);
    
end