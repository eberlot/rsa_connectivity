function varargout = rsa_conn(what,varargin)

baseDir = '/Users/eberlot/Documents/Data/rsa_connectivity';
% example RDM distance matrix
load(fullfile(baseDir,'RDM.mat'));

switch(what)
    
    case 'simulateData'
        mu=1;
        sigma=0.3;
        numCond=5;
        numVox=1000;
        numPart=8;
        numDataset=2;
        numSim=1000;
        signal=1;
        noise = [0:0.1:1];
        trueCorr = [0:0.1:0.9]; % 0 for random, 1 for sameRDM / sameRDM+noise, flexible for specificRDMcorr
        simuType='random'; % what type of simulation: random, sameRDM, sameRDM+noise, specificRDMcorr
        rdmType = {'allPart','averagePart','crossval'}; % what type of rdm calculation
        vararginoptions(varargin,{'mu','sigma','numCond','numVox','numPart','numDataset','numSim','rdmType','simuType','noise','signal','trueCorr'});
        
        condVec    = repmat([1:numCond]',numPart,1);    
        partVec    = kron([1:numPart]',ones(numCond,1));
        NN=[]; SS=[];
        for rt = 1:length(rdmType);
            for t = 1:length(trueCorr);
                for n=1:size(noise,2);
                    for s=1:numSim
                        
                        % first generate random data
                        switch simuType
                            case 'random'
                                data        =  genRandData(mu,sigma,numCond,numVox,numPart,numDataset);
                            case 'sameRDM'
                                data        = genRDMData(D,numCond,numVox,numPart,numDataset);
                            case 'sameRDM+noise'
                                data = genRDMNoisyData(D,numCond,numVox,numPart,numDataset,signal,noise(n));
                            case 'specificRDMcorr'
                                % create rdms with specified correlation
                                rdmGen = calcRDMs_specificCorr(numCond,numDataset,trueCorr(t));
                                for dt = 1:numDataset
                                    data(dt) = genRDMData(pcm_makePD(rdmGen{dt}),numCond,numVox,numPart,1,signal);
                                end
                        end
                        
                        %which type of RDM to calculate
                        switch rdmType{rt}
                            case 'averagePart'
                                rdm = makeRDM_average(data,condVec);
                            case 'allPart'
                                rdm = makeRDM_allPart(data);
                            case 'crossval'
                                rdm = makeRDM_crossval(data,partVec,condVec);
                        end
                        % calculate distances of two datasets
                        N.corr(1,:) = rsa_calcCorrRDMs(rdm);       % rsa correlation
                        N.corr(2,:) = rsa_calcDistCorrRDMs(rdm);    % distance correlation
                        N.corrType  = [1;2];
                        N.numSim    = [s;s];
                        N.noise     = [noise(n);noise(n)];
                        N.trueCorr  = [trueCorr(t);trueCorr(t)];
                        N.rdmType   = [rt;rt];
                        NN=addstruct(NN,N);
                        
                    end
                    % calculate metrics
                    if ~strcmp(simuType,'random')
                        corr_rsa    = NN.corr(NN.corrType==1 & NN.noise==noise(n) & NN.trueCorr==trueCorr(t) & NN.rdmType==rt);
                        corr_dist   = NN.corr(NN.corrType==2 & NN.noise==noise(n) & NN.trueCorr==trueCorr(t) & NN.rdmType==rt);
                        S           = calcSummMetrics(corr_rsa,corr_dist,trueCorr(t));
                        S.trueCorr  = [trueCorr(t);trueCorr(t)];
                        S.noise     = [noise(n);noise(n)];
                        S.rdmType   = [rt;rt];
                        SS=addstruct(SS,S);
                    end
                    % plot histogram of all estimated correlations
                    %hist_corr(N.corr_rsa,N.corr_dist);
                end % noise
            end % trueCorrelation
        end % rdmType
        
        % save the structure
        save(fullfile(baseDir,sprintf('simulations_%s',simuType)),'-struct','NN');
        if ~strcmp(simuType,'random')
            save(fullfile(baseDir,sprintf('simulations_summary_%s',simuType)),'-struct','SS');
        end
    case 'simulateData_new'
        numCond=5;
        numVox=1000;
        numPart=8;
        numDataset=2;
        numSim=1000;
        signal=1;
        theta=1;
        noise = [0:0.1:1];
        simuType='random'; % what type of simulation: random, sameRDM, sameRDM+noise, specificRDMcorr
        rdmType = {'allPart','averagePart','crossval'}; % what type of rdm calculation
        vararginoptions(varargin,{'theta','numCond','numVox','numPart','numDataset','numSim','rdmType','simuType','noise','signal'});
        
        NN=[]; SS=[];
        for rt = 1:length(rdmType);
            for n=1:size(noise,2);
                for s=1:numSim      
                    % first generate random data
                    S.numPart = numPart;
                    S.numVox  = numVox;
                    
                    for d=1:numDataset
                        M{d}.type       = 'feature';
                        M{d}.numGparams = 1;
                        switch simuType % what type of RDM to use as a model
                            case 'sameRDM'
                                Dr{d}   = D;
                                M{d}.Ac = Dr{d}; % use the distance matrix provided
                            case 'random'
                                Dr{d} = rand(numCond);
                                % set diagonals to 0
                                Dr{d}(eye(size(Dr{d}))==1)=0;
                                M{d}.Ac = Dr{d};
                        end
                        [data(d),partVec,condVec] = pcm_generateData(M{d},theta,S,1,signal,noise(n));
                    end
                    % calculate true correlation from inital distances
                    trueCorr(1,:) = rsa_calcCorrRDMs(Dr);
                    trueCorr(2,:) = rsa_calcDistCorrRDMs(Dr);
                     
                    %which type of RDM to calculate
                    switch rdmType{rt}
                        case 'averagePart'
                            rdm = makeRDM_average(data,condVec);
                        case 'allPart'
                            rdm = makeRDM_allPart(data);
                        case 'crossval'
                            rdm = makeRDM_crossval(data,partVec,condVec);
                    end
                    % calculate distances of two datasets
                    N.corr(1,:) = rsa_calcCorrRDMs(rdm);       % rsa correlation
                    N.corr(2,:) = rsa_calcDistCorrRDMs(rdm);    % distance correlation
                    N.corrType  = [1;2];
                    N.numSim    = [s;s];
                    N.noise     = [noise(n);noise(n)];
                    N.trueCorr  = trueCorr;
                    N.rdmType   = [rt;rt];
                    NN=addstruct(NN,N);     
                end
                % calculate metrics
                if ~strcmp(simuType,'random')
             %       corr_rsa    = NN.corr(NN.corrType==1 & NN.noise==noise(n) & NN.rdmType==rt);
             %       corr_dist   = NN.corr(NN.corrType==2 & NN.noise==noise(n) & NN.rdmType==rt);
             %       S           = calcSummMetrics(corr_rsa,corr_dist,trueCorr(t));
             %       S.trueCorr  = [trueCorr(t);trueCorr(t)];
             %       S.noise     = [noise(n);noise(n)];
             %       S.rdmType   = [rt;rt];
             %       SS=addstruct(SS,S);
                end
            end % noise
        end % rdmType
        keyboard;
        % save the structure
        save(fullfile(baseDir,sprintf('simulations_%s',simuType)),'-struct','NN');
        if ~strcmp(simuType,'random')
    %        save(fullfile(baseDir,sprintf('simulations_summary_%s',simuType)),'-struct','SS');
        end
    case 'specificRDMcorr'
        % create two RDM matrices with a specific correlation
        rdmCorr = 0.5;
        numCond = 5;
        numRDMs = 2;
        numSim  = 1;
        numPart = 8;
        numVox = 1000;
        signal = 1;
        noise = 0.1;
        simuType = 'sameRDM';
        rdmType = {'allPart','averagePart','crossval'};
        vararginoptions(varargin,{'rdmCorr','num_offDiag','numRDMs','numSim','numSim','rdmType','simuType','noise'});
        
        TT=[]; SS=[];
        condVec    = repmat([1:numCond]',numPart,1);
        partVec    = kron([1:numPart]',ones(numCond,1));
        
        for rt = 1:length(rdmType);
            for s = 1:size(rdmCorr,2)
                for t = 1:numSim
                    % create rdms with specified correlation
                    rdmGen = calcRDMs_specificCorr(numCond,numRDMs,rdmCorr(t));
                    % now construct data for all different RDMs
                    for i = 1:numRDMs
                        switch simuType
                            case 'sameRDM'
                                data = genRDMData(rdmGen{i},numCond,numVox,numPart,1);
                            case 'sameRDM+noise'
                                data = genRDMNoisyData(rdmGen{i},numCond,numVox,numPart,1,signal,noise);
                        end
                        data = genRDMData(rdmGen{i},numCond,numVox,numPart,1);
                        %which type of RDM to calculate
                        switch rdmType{rt}
                            case 'averagePart'
                                rdmEst(i) = makeRDM_average(data,condVec);
                            case 'allPart'
                                rdmEst(i) = makeRDM_allPart(data);
                            case 'crossval'
                                rdmEst(i) = makeRDM_crossval(data,partVec,condVec);
                        end
                    end   
                    % now calculate correlations
                    T.corr(1,:)    = rsa_calcCorrRDMs(rdmEst); % rsa
                    T.corr(2,:)    = rsa_calcDistCorrRDMs(rdmEst); % dist
                    T.corrType     = [1;2];
                    T.trueCorr     = [rdmCorr(t);rdmCorr(t)];
                    T.numSim       = [s;s];
                    T.rdmType      = [rt;rt];
                    TT=addstruct(TT,T);
                end
                % do summary statistics
                corr_rsa    = TT.corr(TT.corrType==1 & TT.rdmType==rt & TT.trueCorr==rdmCorr(s));
                corr_dist   = TT.corr(TT.corrType==2 & TT.rdmType==rt & TT.trueCorr==rdmCorr(s));
                trueCorr    = TT.trueCorr(TT.rdmType==rt);
                S           = calcSummMetrics(corr_rsa,corr_dist,trueCorr(1));
                S.rdmType   = [rt; rt];
                SS=addstruct(SS,S);
            end            
        end

        
        keyboard;
        % save variables
        save(fullfile(baseDir,'simulations_specificRDMcorr'),'-struct','TT');
        save(fullfile(baseDir,'simulations_summary_specificRDMcorr'),'-struct','SS');
        
        figure
        for r = 1:length(rdmType)
            subplot(1,length(rdmType),r);
            a=getrow(TT,TT.rdmType==r);
            plt.scatter(a.corr(a.corrType==1),a.corr(a.corrType==2));
            drawline(0,'dir','horz'); drawline(0,'dir','vert');
            xlabel('RSA correlation');
            ylabel('Distance correlation');
            title(sprintf('%s',rdmType{r}));
        end
       
        figure
        for r = 1:length(rdmType)
            subplot(1,length(rdmType),r);
            plt.scatter(TT.trueCorr,TT.corr,'subset',TT.rdmType==r & TT.corrType==2);
            hold on;
            meanDist = pivottable(TT.trueCorr,[],TT.corr,'mean','subset',TT.rdmType==1 & TT.corrType==2);
            scatterplot(unique(TT.trueCorr),meanDist,'markercolor','r','markerfill','r','markersize',8);
            drawline(0,'dir','horz'); drawline(0,'dir','vert');
            xlabel('True correlation');
            ylabel('Distance correlation');
            title(sprintf('%s',rdmType{r}));
        end
        
         figure
        for r = 1:length(rdmType)
            subplot(1,length(rdmType),r);
            hold on;
            plt.scatter(TT.trueCorr,TT.corr,'subset',TT.rdmType==r & TT.corrType==1);
            meanDist = pivottable(TT.trueCorr,[],TT.corr,'mean','subset',TT.rdmType==1 & TT.corrType==1);
            scatterplot(unique(TT.trueCorr),meanDist,'markercolor','r','markerfill','r','markersize',8);
            drawline(0,'dir','horz'); drawline(0,'dir','vert');
            xlabel('True correlation');
            ylabel('RSA correlation');
            title(sprintf('%s',rdmType{r}));
        end
    
    case 'RDMcomb'
        numCond = 5;
        corrInit = 0.1; % initial correlation between the pair
        theta = 1;
        numPart = 8;
        numVox = 1000;
        numSim = 1;
        signal = 1;
        noise = 0;
        rdmType = {'allPart','averagePart','crossval'};
        vararginoptions(varargin,{'numCond','numPart','numVox','numSim','theta','corrInit','noise','rdmType'});
        clear D;
      %  D = calcRDMs_specificCorr(numCond,2,corrInit); % calculate starting RDMs
        D{1} = rand(5); 
        D{2} = rand(5);
        % set diagonals to 0
         D{1}(eye(size(D{1}))==1)=0;
         D{2}(eye(size(D{2}))==1)=0;
      % prepare for simulations
      S.numPart = numPart;
      S.numVox  = numVox;
      % calculate a third RDM, make calculations on the three
      w1 = [1:-0.1:0];
      w2 = [0:0.1:1];
      
      RR=[];
      for s = 1:20
          for i = 1:size(w1,2)
              D{3} = D{1}.*w1(i) + D{2}.*w2(i);
              %generate data from the three distances
              for d=1:3
                  M{d}.type                   = 'feature';
                  M{d}.numGparams             = 1;
                  M{d}.Ac                     = D{d}; % use the distance matrix
                  [data(d),partVec,condVec]   = pcm_generateData(M{d},theta,S,numSim,signal,noise);
              end
              
              for r = 1:length(rdmType);  
                  switch rdmType{r}
                      case 'averagePart'
                          rdmEst = makeRDM_average(data,condVec);
                      case 'allPart'
                          rdmEst = makeRDM_allPart(data);
                      case 'crossval'
                          rdmEst = makeRDM_crossval(data,partVec,condVec);
                  end
                  R.rsa_corr  = rsa_calcCorrRDMs(rdmEst);
                  R.dist_corr = rsa_calcDistCorrRDMs(rdmEst);
                  R.sample    = i;
                  R.numSim    = s;
                  R.rdmType   = r;
                  RR=addstruct(RR,R);
              end
          end
      end   
      
          sty1 = style.custom({'red'});
          sty2 = style.custom({'blue'});
          for r=1:length(rdmType)
              figure
              subplot(1,2,1)
              plt.scatter(RR.sample,RR.rsa_corr(:,2),'style',sty1,'subset',RR.rdmType==r);
              hold on
              plt.scatter(RR.sample,RR.rsa_corr(:,3),'style',sty2,'subset',RR.rdmType==r);
              drawline(RR.rsa_corr(1,1),'dir','hor');
              ylabel('RSA corr');
              title(rdmType{r});
              subplot(1,2,2)
              plt.scatter(RR.sample,RR.dist_corr(:,2),'style',sty1,'subset',RR.rdmType==r);
              hold on
              plt.scatter(RR.sample,RR.dist_corr(:,3),'style',sty2,'subset',RR.rdmType==r);
              drawline(RR.dist_corr(1,1),'dir','hor');
              ylabel('Dist corr');
              title(rdmType{r});
          end
    case 'plotSimMetrics'
        simuType='sameRDM+noise'; % what type of simulation: sameRDM, sameRDM+noise, specificRDMcorr
        rdmType={'allPart','averagePart','crossval'};
        corrLabels={'corr-rsa','corr-dist'};
        xVar = 'trueCorr'; % trueCorr or noise
        vararginoptions(varargin,{'simuType','rdmType','xVar'});
        
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
        DD.varD = eval(sprintf('DD.%s',xVar));
        if ~strcmp(simuType,'random');
            SS = load(fullfile(baseDir,sprintf('simulations_summary_%s',simuType)));
            SS.varS = eval(sprintf('SS.%s',xVar));
        end
        
        for s=1:size(rdmType,2);
            D = getrow(DD,DD.rdmType==s);
            
            figure
            plt.scatter(D.varD,D.corr,'split',D.corrType,'leg',corrLabels,'leglocation','northeast');
            xlabel(xVar); ylabel('Estimated correlation');
            title(sprintf('simulation %s - %s RDM',simuType,rdmType{s}));
            
            if ~strcmp(simuType,'random');
                S = getrow(SS,SS.rdmType==s);
                figure
                subplot(2,2,1)
                plt.scatter(S.varS,S.corrEst,'split',S.corrType,'leg',corrLabels,'leglocation','northeast');
                xlabel(xVar); ylabel('Estimated correlation');
                title(sprintf('simulation %s - %s RDM',simuType,rdmType{s}));
                
                subplot(2,2,2)
                plt.scatter(S.varS,S.bias,'split',S.corrType,'leg',corrLabels,'leglocation','northeast');
                xlabel(xVar); ylabel('Bias from true correlation');
                title(sprintf('simulation %s - %s RDM',simuType,rdmType{s}));
                
                subplot(2,2,3)
                plt.scatter(S.varS,S.var,'split',S.corrType,'leg',corrLabels,'leglocation','northeast');
                xlabel(xVar); ylabel('Variance in estimation');
                title(sprintf('simulation %s - %s RDM',simuType,rdmType{s}));
                
                subplot(2,2,4)
                plt.scatter(S.varS,S.mse,'split',S.corrType,'leg',corrLabels,'leglocation','northeast');
                xlabel(xVar); ylabel('MSE');
                title(sprintf('simulation %s - %s RDM',simuType,rdmType{s}));
            end
            
        end
    case 'plotSimHist'
        simuType='sameRDM+noise';
        rdmType={'allPart','averagePart','crossval'};
        corrLabels={'corr-rsa','corr-dist'};
        noiseLevels=[0,0.5,1];
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
          for s=1:size(rdmType,2);
           D = getrow(DD,DD.rdmType==s); 
            
            figure
            for n=1:size(noiseLevels,2)
                d=getrow(D,D.noise==noiseLevels(n));
                subplot(1,size(noiseLevels,2),n)
                tLabel = sprintf('Corr noise-%d %s',noiseLevels(n),rdmType{s});
                hist_corr(d.corr,d.corrType,'legLabels',corrLabels,'titleLabel',tLabel);          
            end
            
          end
    case 'plot_distType'
        simuType='random';
        rdmType={'allPart','averagePart','crossval'};
        noiseLevels=[0,0.5,1];
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
          for s=1:size(rdmType,2);
           D = getrow(DD,DD.rdmType==s); 
            
           figure
           for n=1:size(noiseLevels,2)
               d=getrow(D,D.noise==noiseLevels(n));
               subplot(1,size(noiseLevels,2),n)
               plt.scatter(d.corr(d.corrType==1),d.corr(d.corrType==2));
               xlabel('corr-rsa'); ylabel('corr-dist');
               title(sprintf('Corr noise-%d %s',noiseLevels(n),rdmType{s}));
               ylim([-0.1 1]);
               drawline(0,'dir','horz');
               drawline(0,'dir','vert');
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
     
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions
function data  = genRandData(mu,sigma,numCond,numVox,numPart,numDataset)
% function data = genRandData(mu,sigma,numCond,numVox,numPart,numDataset)
% makes random data for each dataset
    for setNum = 1:numDataset
        data{setNum} = rsa_conn_genData(numCond,numVox,numPart,'mu',mu,'sigma',sigma);
    end
end
function data  = genRDMData(D,numCond,numVox,numPart,numDataset,signal)
% function data = genRDMData(D,numCond,numVox,numPart,numDataset,signal)
% makes data from a sample RDM - multivariate normal 
    for setNum = 1:numDataset
        if iscell(D)
            RDM = D{setNum};
        else
            RDM = D;
        end
        % get the mean and covariance from provided distance matrix D
        M{setNum}.type                   = 'feature';
        M{setNum}.numGparams             = 1;
        M{setNum}.Ac                     = RDM; % use the distance matrix
        
        data{setNum}   = pcm_generateData(M{d},theta,S,numSim,signal,noise);
        data{setNum} = rsa_conn_genData(numCond,numVox,numPart,'dataType','fromRDM','RDM',pcm_makePD(RDM),'signal',signal);
    end
end
function data  = genRDMNoisyData(D,numCond,numVox,numPart,numDataset,signal,noise)
% function data  = genSameNoisyData(D,numCond,numVox,numPart,numDataset,signal,noise)
% generate data from same RDM with added noise
    for setNum = 1:numDataset
        data{setNum} = rsa_conn_genData(numCond,numVox,numPart,'RDM',D,'dataType','fromRDM+noise','signal',signal,'noise',noise);
    end
end
function rdm   = makeRDM_average(data,condVec)
% function rdm = makeRDM_average(data,condVec)
% makes RDM matrix from euclidean distances (not crossvalidated)
    numDataset=size(data,2);
    % make indicator matrix for estimating mean pattern
    X = indicatorMatrix('identity_p',condVec);
    for st=1:numDataset
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm{st}=rsa_squareRDM(pdist(D{st}));
    end; 
end
function rdm   = makeRDM_allPart(data)
% function rdm = makeRDM_allPart(data)
% makes RDM matrix from euclidean distances across all runs (not averaging)
    numDataset=size(data,2);
    % make indicator matrix for estimating mean pattern 
    % one factor for each condition & partition
    X = indicatorMatrix('identity_p',1:size(data{1},1));
    for st=1:numDataset
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm{st}=rsa_squareRDM(pdist(D{st}));
    end;
end
function rdm   = makeRDM_crossval(data,partVec,condVec)
% function rdm   = makeRDM_crossval(data,partVec,condVec)
% makes crossvalidated RDM matrix
numDataset=size(data,2);
for st=1:numDataset
    % calculate crossvalidated squared Euclidean distances
    rdm{st}=rsa_squareRDM(rsa.distanceLDC(data{st},partVec,condVec));
end;

end
function S     = calcSummMetrics(corr_rsa,corr_dist,trueCorr)
% function sumM = calcMetrics(corr_rsa,corr_dist,trueCorr)
% calculates summary metrics for estimated rsa and distance correlations

    S.corrEst   = [mean(corr_rsa);mean(corr_dist)];
    S.bias      = [trueCorr-mean(corr_rsa);trueCorr-mean(corr_dist)];
    S.var       = [var(corr_rsa);var(corr_dist)];
    S.mse       = [sum(trueCorr-corr_rsa).^2;sum(trueCorr-corr_dist).^2];
    S.corrType  = [1;2];
    S.trueCorr  = [trueCorr; trueCorr];

end
function hist_corr(corr,corrType,varargin)
% function hist_corr(corr_rsa,corr_dist,vararginoptions)
% plots the distribution of rsa and distance correlations
% varargin - optional level of noise for the title
vararginoptions(varargin,{'legLabels','titleLabel'});

    plt.hist(corr,'split',corrType,'leg',legLabels,'leglocation','northeast');
    drawline(mean(corr(corrType==1)),'dir','vert','color','b');
    drawline(mean(corr(corrType==2)),'dir','vert','color','g');
    xlabel('Estimated correlation');
    title(titleLabel);
    
end