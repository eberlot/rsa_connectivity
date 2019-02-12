function varargout = rsa_connect(what,varargin)

baseDir = '/Volumes/MotorControl/data/rsa_connectivity';
%baseDir = cd;

legLabel = {'RDM-corr','cRDM-cos','cRDM-cos-sqrt','cRDM-cos-uniprewh','cRDM-cos-multiprewh','multiDepend'};
% for plotting
gray=[80 80 80]/255;
lightgray=[160 160 160]/255;
lightlightgray=[200 200 200]/255;
silver=[240 240 240]/255;
black=[0 0 0]/255;
blue=[49,130,189]/255;
%lightblue=[158,202,225]/255;
red=[222,45,38]/255;
lightred=[252,146,114]/255;
sAll = style.custom({blue,black,gray,lightgray,silver,red});

%c1=[39 38 124]/255;
c2=[140 140 185]/255;
c3=[249 191 193]/255;
c1=[0 0 200]/255;
sBW = style.custom({black,gray,lightgray,silver});
sB  = style.custom({c1,c2,c3});
styTrio = style.custom({black,gray,lightgray},'markertype',{'o','v','s'},'linestyle',{'-','--','-.'});

switch(what)

    case 'run_simulation'
        nCond = 5;
        nPart = 8;
        nVox = 100;
        numSim = 100;
        varReg = [0,1,5,10:5:30];
        corrReg = 0:0.1:0.9;
        signal = 1;
        type = 11; 
        noiseType='within';
        vararginoptions(varargin,{'nCond','numSim','RDMtype','corrRDM','type','noiseType','distCalc','signal','nVox'});
        NN=[];
        
        switch noiseType
            case 'within'
                varReg = [0.1:0.1:1,2:1:10,12:2:20];
                corrReg = 0;
                numSim = 500;
            case 'within_oneLevel'
                varReg  = 4;
                corrReg = 0;
                numSim = 5000;
            case 'both'
                varReg = [1,10,40];
                corrReg = [0:0.1:0.9,0.99,0.999];
            case 'between'
                varReg  = [0.1,0.5,1,4,8,12];
                corrReg = 0:0.1:0.9;
                numSim = 500;
        end
        [G,D] = makeGs(nCond,type);
        %  prepare model
        nRDM = size(G,2);
        trueRDM = zeros(nRDM,nCond*(nCond-1)/2);
        M = cell(1,nRDM);
        for i=1:nRDM
            trueRDM(i,:)=rsa_vectorizeRDM(D{i});
            M{i}=makeModel('sameRDM',G{i},nCond);
        end
        % other details for data generation
        S.numPart = nPart;
        S.numVox  = nVox;
        % calculate true distances
        [T.trueDist,T.reg1,T.reg2,T.distType] = calcTrueDist(trueRDM);
        for r=corrReg
            for v=varReg
                for n=1:numSim
                    data = cell(1,nRDM);
                    V    = cell(1,nRDM);
                    for i=1:nRDM
                        [data{i},partVec,condVec] = makePatterns(M{i}.Gc,'signal',signal,'nPart',S.numPart,'nVox',S.numVox); %signal 1, noise 0
                        V{i} = zeros(size(trueRDM,2));
                    end
                    % add shared noise across regions
                    data = addSharedNoise(data,v,r);
                    for distType=1:6
                        if distType==4
                            for i=1:nRDM
                                [V{i},~,~]=covariance_dist(partVec,condVec,'G',M{i}.Gc,'sigma',eye(nCond)*v,'nVox',nVox);
                            end
                        end
                        % splithalf or not
                        % calculate RDMs, distances between them
                        [N.calcDist,N.reg1,N.reg2,N.distType] = calcDistAll(data,partVec,condVec,distType,V);
                        [N.trueDist] = calcTrueDistAll(trueRDM,distType,V);
                        sizeAll      = size(N.calcDist,1);
                        N.varReg     = repmat(v,sizeAll,1);
                        N.corrReg    = repmat(r,sizeAll,1);
                        N.covReg     = repmat(r*v,sizeAll,1);
                      %  N.dataCorr   = dataCorr;
                        N.numSim     = repmat(n,sizeAll,1);
                        % save Vs
                    %    N.V(1,:)     = rsa_vectorizeIPMfull(V{2});
                    %    N.V(2,:)     = rsa_vectorizeIPMfull(V{3});
                    %    N.V(3,:)     = rsa_vectorizeIPMfull(V{4});
                      %  N.V          = repmat({V},sizeAll,1);
                        NN=addstruct(NN,N);
                    end
                end; % number of simulations
                fprintf('%d.',find(v==varReg));
            end; % within-reg
            fprintf('\nDone %d/%d\n',find(r==corrReg),numel(corrReg));
        end; % across-reg
        
        save(fullfile(cd,sprintf('dist_noise_%s_type%d', noiseType,type)),'-struct','NN');
        save(fullfile(cd,sprintf('truth_noise_%s_type%d',noiseType,type)),'T','trueRDM');
    case 'plot_within'
        noiseType='within';
        type=11;
        vararginoptions(varargin,{'type'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('dist_noise_%s_type%d', noiseType,type)));
        load(fullfile(baseDir,sprintf('truth_noise_%s_type%d',noiseType,type)));
        
        legend = {'RDM-corr-reg2','cRDM-corr-reg2','cRDM-split-corr-reg2',...
            'RDM-corr-reg3','cRDM-corr-reg3','cRDM-split-corr-reg3',...
            'RDM-corr-reg4','cRDM-corr-reg4','cRDM-split-corr-reg4'};
        keyboard;
        D=getrow(D,ismember(D.varReg,0:1:20));
        figure
        subplot(311)       
        color     = {c1,c1,c1,c2,c2,c2,c3,c3,c3};
        linestyle   = {'-','--','-.'};
        markertype  = {'o','v','^'};

        lineplot(D.varReg,D.calcDist,'subset',D.distType<4,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
            'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
            'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(1),'dir','horz','color',c1);
        drawline(T.trueDist(2),'dir','horz','color',c2);
        drawline(T.trueDist(3),'dir','horz','color',c3);
        title('correlation distances');

        subplot(312)
        color     = {c1,c1,c2,c2,c3,c3};
        linestyle   = {'--','-.'};
        markertype  = {'v','^'};
        legend = {'cRDM-cosine-reg2','cRDM-split-cosine-reg2',...
            'cRDM-cosine-reg3','cRDM-split-cosine-reg3',...
            'cRDM-cosine-reg4','cRDM-split-cosine-reg4'};         
        lineplot(D.varReg,D.calcDist,'subset',D.distType>3&D.distType<6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
                     'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(4),'dir','horz','color',c1);
        drawline(T.trueDist(5),'dir','horz','color',c2);
        drawline(T.trueDist(6),'dir','horz','color',c3);
        title('cosine distances');
        
        subplot(313)
        legend = {'multiDepend-reg2','multiDepend-reg3','multiDepend-reg4'};
        
        color     = {c1,c2,c3};
        linestyle   = {'-.'};
        markertype  = {'^'};
        lineplot(D.varReg,D.calcDist,'subset',D.distType==6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
                     'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
         xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        title('multivariate dependence (Anzellotti)');
    case 'plot_dist'
        noiseType='within';
        type=1;
        noiseL = [0.1,0.5,1,5];
        distType = 'trueDist'; % trueDist or calcDist
        distLab = {'corr','cos','ssqrt-cos','uniprewh-cos','multiprewh-cos','noiseL'};
        vararginoptions(varargin,{'type','noiseL','distType'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('dist_noise_%s_type%d', noiseType,type)));
        load(fullfile(baseDir,sprintf('truth_noise_%s_type%d',noiseType,type)));
 
        for v=noiseL
            figure
            for d=1:5 % distType
                subplot(5,1,d)
                plt.hist(D.(distType),'split',D.reg2,'subset',D.varReg==v & D.distType==d);
                hold on;
                drawline(mean(D.(distType)(D.varReg==v & D.distType==d & D.reg2==2)),'dir','vert','color',[0 0 1]);
                drawline(mean(D.(distType)(D.varReg==v & D.distType==d & D.reg2==3)),'dir','vert','color',[0 1 0]);
                drawline(mean(D.(distType)(D.varReg==v & D.distType==d & D.reg2==4)),'dir','vert','color',[1 0 0]);
                if d==1
                    title(sprintf('%s - noise %2.1f',distLab{d},v));
                else
                    title(sprintf('%s',distLab{d}));
                end
            end
        end
    case 'prewh_level2'
        % plot behaviour of uni prewhitened cosine dist in relation to sqrt
        % and squared
        noiseType='within';
        type=11;
        vararginoptions(varargin,{'type','noiseL'});
        % choose style
        load(fullfile(baseDir,sprintf('truth_noise_%s_type%d',noiseType,type)));
        D=load(fullfile(baseDir,sprintf('dist_noise_%s_type%d', noiseType,type)));
        D = getrow(D,D.reg1==1);
        noiseL = unique(D.varReg);
        KK=[];
        for v=noiseL'
            T = getrow(D,D.varReg==v & ismember(D.distType,[2,3,4]));
            t1 = T.trueDist(T.reg2==2);
            t2 = T.trueDist(T.reg2==3);
            t3 = T.trueDist(T.reg2==4);
            
            K.distRatio = t2./(t3-t1);
            K.distRatio = t2;
            K.distType = T.distType(T.reg2==2);
            K.noiseL = ones(size(K.distType))*v;
            KK=addstruct(KK,K);
        end
        figure
        subplot(231)
        imagesc(rsa_squareRDM(trueRDM(1,:)));
        colormap('hot');
        subplot(232)
        imagesc(rsa_squareRDM(trueRDM(3,:)));
        colormap('hot');
        subplot(233)
        imagesc(rsa_squareRDM(trueRDM(4,:)));
        colormap('hot');
        subplot(2,3,4:6)
        plt.line(KK.noiseL,KK.distRatio,'split',KK.distType,'leg',{'squared','sqrt','uni'},'style',styTrio);
        ylabel('Normalized alpha d2');
        xlabel('noise');  
    case 'prewh_level1'
        nCond = 5;
        nPart = 8;
        nVox = 100;
        numSim = 100;
        signal = 1;
        type=1;
        vararginoptions(varargin,{'nCond','numSim','RDMtype','corrRDM','type','noiseType','distCalc','signal','nVox'});
        NN=[];
        
        varReg = [0.1:0.1:1,2:1:30];
        [G,D] = makeGs(nCond,type);
        %  prepare model
        nRDM = size(G,2);
        trueRDM = zeros(nRDM,nCond*(nCond-1)/2);
        M = cell(1,nRDM);
        for i=1:nRDM
            trueRDM(i,:)=rsa_vectorizeRDM(D{i});
            M{i}=makeModel('sameRDM',G{i},nCond);
        end
        % other details for data generation
        S.numPart = nPart;
        S.numVox  = nVox;
        for v=varReg
            for n=1:numSim
                for i=1:nRDM
                    [data{1},partVec,condVec] = makePatterns(M{i}.Gc,'signal',signal,'nPart',S.numPart,'nVox',S.numVox); %signal 1, noise 0
                    % add shared noise across regions
                    data = addSharedNoise(data,v,0,nCond);
                    [V{1},~,~]=covariance_dist(partVec,condVec,'G',M{i}.Gc,'sigma',eye(nCond)*v,'nVox',nVox);
                    % here make RDMs
                    D1=trueRDM(i,:); % squared
                    D2=ssqrt(D1); % sqrt
                    D3=rdm_prewhitenDist(D1,V,'type','uni'); % uniprewh
                    %D4=rdm_prewhitenDist(D1,V,'type','multi'); % uniprewh
                    N.trueDist  = calcDist([D1;D2;D3],'cosine');
                    N.RDMtype   = [1;2;3];
                    N.RDM1      = [1;1;3];
                    N.RDM2      = [2;3;3];
                    N.numRDM    = [i;i;i];
                    N.varReg    = [v;v;v];
                    N.numSim    = [n;n;n];
                    
                    NN=addstruct(NN,N);
                end
            end; % number of simulations
            fprintf('%d.',find(v==varReg));
        end; % within-reg
        figure 
        plt.line(NN.varReg,NN.trueDist,'split',NN.RDMtype,'subset',NN.numRDM==4,'leglocation','east','leg',{'squared-sqrt','squared-uni','sqrt-uni'},'style',styTrio);
        hold on;
        drawline(0,'dir','horz');
        xlabel('noise level');
        ylabel('distance between RDMs');
        
    case 'plot_performance'
        noiseType='within';
        type=1;
        vararginoptions(varargin,{'type','noiseType'});
        % choose style
        color     = {blue,black,gray,lightgray,lightlightgray,red};
        linestyle   = {'-','-','-.','--','-','-'};
        markertype  = {'o','o','^','v','s','o'};
        switch noiseType
            case 'within'
                metric='varReg';
            case 'within_oneLevel'
                metric='varReg';
            case 'between'
                metric='corrReg';
        end
        
        D=load(fullfile(baseDir,sprintf('dist_noise_%s_type%d', noiseType,type)));
        D = getrow(D,D.reg1==1); % only comparisons to reg1=1
        load(fullfile(baseDir,sprintf('truth_noise_%s_type%d',noiseType,type)));
        CC=[];
        for i=unique(D.(metric))'
            for j=1:max(D.distType)
                T=getrow(D,D.(metric)==i & D.distType==j);
                dist1=T.calcDist(T.reg2==2);
                dist2=T.calcDist(T.reg2==3);
                dist3=T.calcDist(T.reg2==4);
                C.error=1-sum(dist1<dist2 & dist2<dist3)/length(dist1);
                C.decision1=sum(dist1<dist2)/length(dist1);
                C.decision2=sum(dist2<dist3)/length(dist1);
                C.distType=j;
                C.(metric)=i;
                CC=addstruct(CC,C);
            end
        end
        figure
        subplot(341)
        imagesc(rsa_squareRDM(trueRDM(1,:)));
        subplot(342)
        imagesc(rsa_squareRDM(trueRDM(2,:)));
        subplot(343)
        imagesc(rsa_squareRDM(trueRDM(3,:)));
        subplot(344)
        imagesc(rsa_squareRDM(trueRDM(4,:)));
        subplot(3,4,5:8)
        switch noiseType
            case 'within'
                lineplot(CC.(metric),CC.error,'split',CC.distType,'style_thickline',...
                    'linecolor',color,'linestyle',linestyle,...
                    'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                    'errorcolor',color,'shadecolor',color,'leg',legLabel);
                xlabel(sprintf('%s region noise',noiseType));
                drawline(5/6,'dir','horz','linestyle','--');
                % title(sprintf('%s',distCalc{c}));
                ylabel('Errors in retrieving structure');
                subplot(3,4,[9,10])
                lineplot(CC.(metric),CC.decision1,'split',CC.distType,'style_thickline',...
                    'linecolor',color,'linestyle',linestyle,...
                    'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                    'errorcolor',color,'shadecolor',color,'leg',legLabel,'leglocation','northeast');
                drawline(0.5,'dir','horz','linestyle','--');
                % title(sprintf('%s',distCalc{c}));
                xlabel(sprintf('%s region noise',noiseType));
                ylabel('Correct decision 1');
                subplot(3,4,[11,12])
                lineplot(CC.(metric),CC.decision2,'split',CC.distType,'style_thickline',...
                    'linecolor',color,'linestyle',linestyle,...
                    'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                    'errorcolor',color,'shadecolor',color,'leg',legLabel,'leglocation','northeast');
                drawline(0.5,'dir','horz','linestyle','--');
                % title(sprintf('%s',distCalc{c}));
                xlabel(sprintf('%s region noise',noiseType));
                ylabel('Correct decision 2');
            case 'within_oneLevel'
                barplot(CC.distType,CC.error);
                set(gca,'XTickLabel',legLabel);
                drawline(5/6,'dir','horz','linestyle','--');
                % title(sprintf('%s',distCalc{c}));
                ylabel('Errors in retrieving structure');
                subplot(3,4,[9,10])
                barplot(CC.distType,CC.decision1);
                ylabel('Correct decision 1');
                subplot(3,4,[11,12])
                barplot(CC.distType,CC.decision2);
                ylabel('Correct decision 2');
        end
        
    %    pivottable(D.distType,D.reg2,D.calcDist,'mean','subset',D.varReg<6);
    case 'plot_performance_between'
        noiseType='between';
        type=1;
        vararginoptions(varargin,{'type','noiseType'});
        % choose style
        legLabel = {'RDM-corr','cRDM-cos','cRDM-cos-sqrt','cRDM-cos-uniprewh','cRDM-cos-multiprewh','multiDepend'};
        
        D=load(fullfile(baseDir,sprintf('dist_noise_%s_type%d', noiseType,type)));
        D = getrow(D,D.reg1==1); % only comparisons to reg1=1
        load(fullfile(baseDir,sprintf('truth_noise_%s_type%d',noiseType,type)));
        CC=[];
        for i=unique(D.corrReg)'
            for v=unique(D.varReg)'
                for j=1:max(D.distType)
                    T=getrow(D,D.corrReg==i & D.varReg==v & D.distType==j);
                    dist1=T.calcDist(T.reg2==2);
                    dist2=T.calcDist(T.reg2==3);
                    dist3=T.calcDist(T.reg2==4);
                    C.error=1-sum(dist1<dist2 & dist2<dist3)/length(dist1);
                    C.decision1=sum(dist1<dist2)/length(dist1);
                    C.decision2=sum(dist2<dist3)/length(dist1);
                    C.distType=j;
                    C.corrReg=i;
                    C.varReg=v;
                    CC=addstruct(CC,C);
                end
            end
        end
        
        idx=1;
        figure
        for t=unique(CC.varReg)'
            subplot(length(unique(CC.varReg)),1,idx);
            plt.line(CC.corrReg,CC.error,'split',CC.distType,'subset',CC.varReg==t,'style',sAll,'leg',legLabel);
            ylabel('error rate');
            idx=idx+1;
        end

    case 'calc_confusability_both'
         noiseType='both';
        type=1;
        vararginoptions(varargin,{'type','noiseType'});
        % choose style
        legLabel = {'RDM-corr','cRDM-cos','cRDM-cos-sqrt','cRDM-cos-prewh','multiDepend'};
        color     = {black,lightgray,lightlightgray,red,lightred,blue};
        linestyle   = {'-','--','-.','--','-.','-.'};
        markertype  = {'o','v','^','v','^','^'};
        
        TT=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_type%d', noiseType,type)));
        %T1=getrow(TT,ismember(TT.varReg,[1.1,10,20,40]));
        T1=getrow(TT,TT.varReg<20);
        figure
        uV = unique(T1.varReg);
        CC=[];
        
        for c=uV'
            D=getrow(T1,T1.varReg==c);
            for i=unique(D.corrReg)'
                for j=1:max(D.distType)
                    T=getrow(D,D.corrReg==i & D.distType==j);
                    dist1=T.calcDist(T.reg2==2);
                    dist2=T.calcDist(T.reg2==3);
                    dist3=T.calcDist(T.reg2==4);
                    C.confus=1-sum(dist1<dist2 & dist2<dist3)/length(dist1);
                    C.distType=j;
                    C.corrReg=i;
                    C.varReg=c;
                    C.dataCorr=nanmean(T.dataCorr);
                    CC=addstruct(CC,C);
                end
            end
        end
        figure
        lineplot(CC.dataCorr,CC.confus,'split',CC.distType,'subset',CC.varReg==18,'style_thickline',...
            'linecolor',color,'linestyle',linestyle,...
            'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
            'errorcolor',color,'shadecolor',color,'leg',legLabel);
        drawline(5/6,'dir','horz','linestyle','--');
        xlabel(sprintf('%s region noise',noiseType));
        ylabel('Confusability');
        figure
        lineplot(CC.corrReg,CC.confus,'split',[CC.distType CC.varReg],'style_thickline',...
            'linecolor',color,'linestyle',linestyle,...
            'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
            'errorcolor',color,'shadecolor',color,'leg',legLabel);
        drawline(5/6,'dir','horz','linestyle','--');
        xlabel(sprintf('%s region noise',noiseType));
        ylabel('Confusability');
    case 'plot_between'
        noiseType='between';
        type=1;
        vararginoptions(varargin,{'type'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_type%d', noiseType,type)));
        load(fullfile(baseDir,sprintf('simulation_truth_noise_%s_type%d',noiseType,type)));
    
        legLabel = {'RDM-corr','cRDM-cos','cRDM-cos-sqrt','cRDM-cos-uniprewh','cRDM-cos-multiprewh','multiDepend'};
        color     = {blue,black,gray,lightgray,lightlightgray,red};
        linestyle   = {'-','-','-.','--','-','-'};
        markertype  = {'o','o','^','v','s','o'};
        
        figure        
        subplot(311)
        lineplot(D.corrReg,D.calcDist,'subset',D.varReg==1&D.reg2==2,'errorfcn','std',...
            'split',[D.reg2 D.distType],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel,'leglocation','northeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        %drawline(T.trueDist(1),'dir','horz','color',c1);
        
        subplot(312)
        lineplot(D.corrReg,D.calcDist,'subset',D.varReg==1&D.reg2==3,'errorfcn','std',...
            'split',[D.reg2 D.distType],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
       % drawline(T.trueDist(4),'dir','horz','color',c1);
       % drawline(T.trueDist(5),'dir','horz','color',c2);
       % drawline(T.trueDist(6),'dir','horz','color',c3);
        
        subplot(313)
        lineplot(D.corrReg,D.calcDist,'subset',D.varReg==1&D.reg2==4,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
    case 'plot_between_multi'
        noiseType='both';
        distCalc='ssqrt';
        vararginoptions(varargin,{'distCalc'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)));
        load(fullfile(baseDir,sprintf('simulation_truth_noise_%s_dist_%s',noiseType,distCalc)));
    
        T = getrow(D,ismember(D.varReg,[1,20]));
        color     = {black,lightlightgray,red,lightred,blue};
        linestyle   = {'-','-','-','-','-','--','--','--','--','--','-.','-.','-.','-.','-.'};
        markertype  = {'o','o','o','o','o','v','v','v','v','v','^','^','^','^','^'};
        
        figure
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & ismember(T.distType,[1,3,4,5,6]),...
            'split',[T.varReg T.distType],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color);
        subplot(132)
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & T.distType==2,'split',T.varReg,'style_thickline');
        subplot(133)
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & T.distType==5,'split',T.varReg,'style_thickline');
        keyboard;
    
    case 'evaluate'
        type=3;
        vararginoptions(varargin,{'type'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        load(fullfile(baseDir,sprintf('simulation_truth_type%d_new',type)));
        
        EE=[];
        range = [2,2,1];
        for s=1:2
            uRDM = unique(D.RDMtype(D.splitHalf==s-1));
            for rdm=1:length(uRDM)
                D1 = getrow(D,D.RDMtype==uRDM(rdm) & D.splitHalf==s-1);
                for i=unique(D1.distType)' % each distType
                    for c1=unique(D1.corrReg)' % correlation
                        for c2=unique(D1.varReg)' % variance
                            % determine true distance
                            if i==1
                                trueDist(1)=T.trueDist(1);
                                trueDist(2)=T.trueDist(2);
                                trueDist(3)=T.trueDist(3);
                            elseif i==2
                                trueDist(1)=T.trueDist(4);
                                trueDist(2)=T.trueDist(5);
                                trueDist(3)=T.trueDist(6);
                            else
                                trueDist(1)=T.trueDist(1);
                                trueDist(2)=T.trueDist(2);
                                trueDist(3)=1;
                            end
                            Dd = getrow(D1,D1.distType==i & D1.corrReg==c1 & D1.varReg==c2);
                            ind1 = Dd.calcDist(Dd.reg2==2);
                            ind2 = Dd.calcDist(Dd.reg2==3);
                            ind3 = Dd.calcDist(Dd.reg2==4);
                            % obtain metrics - only for distance reg2=2
                            % calculate bias
                            tmpTrue = mean(D1.calcDist(D1.distType==i & D1.corrReg==0 & D1.varReg==c2 & D1.reg2==3));
                            E.bias=abs((mean(ind2)-tmpTrue)/range(i));
                            % calculate variance
                            E.var=std(ind2)/range(i);
                            % determine how often correct structure
                            E.confus=sum(ind1<ind2 & ind2 < ind3)/length(ind1);
                            % other info          
                            E.varReg    = c2;
                            E.corrReg   = c1;
                            E.distType  = i;
                            E.splitHalf = s-1;
                            E.RDMtype   = uRDM(rdm);
                            EE=addstruct(EE,E);
                        end
                    end
                end
            end
        end
        save(fullfile(baseDir,sprintf('evaluation_type%d_new',type)),'-struct','EE');
    case 'evaluate_tmp'
        D = load(fullfile(baseDir,'sim_tmp'));
        load(fullfile(baseDir,sprintf('simulation_truth_type3_new')));
        EE=[];
        range = [2,2,1];
        type=unique(D.ind)';
        for i=1:max(type)
            D1=getrow(D,D.ind==i);
            for c1=unique(D1.corrReg)' % correlation
                for c2=unique(D1.varReg)' % variance
                    % determine true distance
                    if i==1 || i==2 || i==6
                        trueDist(1)=T.trueDist(1);
                        trueDist(2)=T.trueDist(2);
                        trueDist(3)=T.trueDist(3);
                        range2=range(1);
                    elseif i==3 || i==4
                        trueDist(1)=T.trueDist(4);
                        trueDist(2)=T.trueDist(5);
                        trueDist(3)=T.trueDist(6);
                        range2=range(2);
                    else
                        trueDist(1)=T.trueDist(1);
                        trueDist(2)=T.trueDist(2);
                        trueDist(3)=1;
                        range2=range(3);
                    end
                    Dd = getrow(D1,D1.corrReg==c1 & D1.varReg==c2);
                    ind1 = Dd.calcDist(Dd.reg2==2);
                    ind2 = Dd.calcDist(Dd.reg2==3);
                    ind3 = Dd.calcDist(Dd.reg2==4);
                    % obtain metrics - only for distance reg2=2
                    % calculate bias
                    E.bias=abs((mean(ind2)-trueDist(2))/range2);
                    % calculate variance
                    % determine how often correct structure
                    E.confus=sum(ind1<ind2 & ind2 < ind3)/length(ind1);
                    % other info
                    E.varReg    = c2;
                    E.corrReg   = c1;
                    E.ind       = i;
                    EE=addstruct(EE,E);
                end
                
            end
        end
        save(fullfile(baseDir,'eval_tmp'),'-struct','EE');
    case 'plot_withinNoise'
        type=1;
        splitHalf=0;
        distType=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.corrReg==0 & D.splitHalf==splitHalf & D.distType==distType);
        figure
        plt.scatter(T1.varReg,T1.calcDist,'subset',T1.reg1==1&T1.reg2==3,'split',T1.RDMtype,'style',sBW);
      % plt.scatter(T1.varReg,T1.calcDist,'subset',T1.reg1==1&T1.reg2==3 & T1.RDMtype==1,'style',sBW);
    case 'plot_withinNoise_histogram'
        type=3;
        splitHalf=0;
        distType=1;
        RDMtype=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.corrReg==0 & D.splitHalf==splitHalf & D.distType==distType & D.RDMtype==RDMtype);
        unVar = unique(T1.varReg);
        for i=unVar'
            figure
            plt.hist(T1.calcDist,'split',T1.reg2,'subset',T1.varReg==i,'style',sB);
            hold on;
            %drawline(mean(T1.calcDist(T1.reg2==2 & T1.varReg==i)),'dir','vert','color',c1);
            %drawline(mean(T1.calcDist(T1.reg2==3 & T1.varReg==i)),'dir','vert','color',c2);
            %drawline(mean(T1.calcDist(T1.reg2==4 & T1.varReg==i)),'dir','vert','color',c3);
        end
    case 'plot_betweenNoise'
        type=1;
        splitHalf=0;
        RDMtype=1;
        vararginoptions(varargin,{'distType','type','split','RDMtype'});

       for r=RDMtype
           D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
         %  D=getrow(D,ismember(D.varReg,[10,30,60,90]) & D.splitHalf==split & D.RDMtype==RDMtype & D.reg1==2 & D.reg2==3);
           D=getrow(D, D.splitHalf==splitHalf & D.RDMtype==RDMtype & D.reg1==1 & D.reg2==4);
           figure
           for v=1:2 % for each distance
               subplot(1,3,v)
               plt.line(D.corrReg,D.calcDist,'split',D.varReg,'style',sBW,'subset',D.distType==v,'style',sBW);
               xlabel('Shared noise');
               ylabel('Estimated distance');
           end
       end
    case 'plot_betweenNoise_histogram'
        type=3;
        splitHalf=0;
        distType=1;
        RDMtype=2;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.varReg==10 & D.splitHalf==splitHalf & D.distType==distType & D.RDMtype==RDMtype);
        cReg = unique(T1.corrReg);
        for i=cReg'
            figure
            plt.hist(T1.calcDist,'split',T1.reg2,'subset',T1.corrReg==i,'style',sB);
            hold on;
            drawline(mean(T1.calcDist(T1.reg2==2 & T1.corrReg==i)),'dir','vert','color',c1);
            drawline(mean(T1.calcDist(T1.reg2==3 & T1.corrReg==i)),'dir','vert','color',c2);
            drawline(mean(T1.calcDist(T1.reg2==4 & T1.corrReg==i)),'dir','vert','color',c3);
        end
    case 'plot_evaluate_within'
        type=3;
        splitHalf=0;
        RDMtype=2;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.corrReg==0 & D.split==splitHalf & (D.RDMtype==RDMtype | D.RDMtype==0));
        figure
        subplot(131)
        plt.line(D.varReg,D.confus,'split',D.distType,'style',sBW);
        ylabel('Confusability');
        xlabel('Within-region noise');
        subplot(132)
        plt.line(D.varReg,D.bias,'split',D.distType,'style',sBW);
        ylabel('Bias');
        xlabel('Within-region noise');
        subplot(133)
        plt.line(D.varReg,D.var,'split',D.distType,'style',sBW);
        ylabel('variance');
        xlabel('Within-region noise');
    case 'plot_evaluate_across_var'
        type=3;
        splitHalf=0;
        RDMtype=2;
        distType=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.split==splitHalf & D.RDMtype==RDMtype & D.distType==distType);
        D=getrow(D,ismember(D.varReg,[10,20,30,60]));
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',D.varReg,'style',sBW);
        ylabel('Correct structure');
        xlabel('within-region noise');
        subplot(132)
        plt.line(D.corrReg,D.bias,'split',D.varReg,'style',sBW);
        ylabel('Bias');
        xlabel('within-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',D.varReg,'style',sBW);
        ylabel('variance');
        xlabel('within-region noise');
    case 'plot_evaluate_across_distType'
        type=3;
        splitHalf=1;
        RDMtype=2;
        varReg=10;
        vararginoptions(varargin,{'type','split','RDMtype','distType','varReg'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.split==splitHalf & (D.RDMtype==RDMtype | D.RDMtype==0) & D.varReg==varReg);
       % D=getrow(D,ismember(D.varReg,[10,30,60,90]));
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',D.distType,'style',sBW);
        ylabel('Confusability');
        xlabel('across-region noise');
        subplot(132)
        plt.line(D.corrReg,D.bias,'split',D.distType,'style',sBW);
        ylabel('Bias');
        xlabel('across-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',D.distType,'style',sBW);
        ylabel('variance');
        xlabel('across-region noise');
        
       % plt.line(D.corrReg,D.var,'split',D.varReg,'subset',D.distType==2);
    case 'plot_evaluate_across'
        type=3;
        RDMtype=2;
        varReg=60;
        vararginoptions(varargin,{'type','split','RDMtype','distType','varReg'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.RDMtype==RDMtype & D.varReg==varReg);
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('Confusability');
        xlabel('within-region noise');
        subplot(132)
        plt.line(D.corrReg,abs(D.bias),'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('Bias');
        xlabel('within-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('variance');
        xlabel('within-region noise');
        
        figure
        t=getrow(D,ismember(D.corrReg,[0.1,0.9]));
        subplot(131)
        plt.bar(t.corrReg,t.confus,'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Confusability');
        subplot(132)
        plt.bar(t.corrReg,abs(t.bias),'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Bias');
        subplot(133)
        plt.bar(t.corrReg,t.var,'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Variance');
        
%         figure
%         subplot(131)
%         plt.bar(t.corrReg,t.confus,'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Correct structure');
%         subplot(132)
%         plt.bar(t.corrReg,abs(t.bias),'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Bias');
%         subplot(133)
%         plt.bar(t.corrReg,t.var,'split',[ t.split t.distType],'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Variance');
        
    case 'KL_div'
        % noiseless example
        % different Gs
        % consider cosine, distanceCorr, euclidean, KL
        condN=5;
        theta=1;
        nPart=1;
        nVox=1000;
        figPlot=1;
        vararginoptions(varargin,{'condN','figPlot'});
        
        % first make Gs of interest
        G = makeGs(condN);
        if figPlot
            figure
            for i=1:6
                subplot(3,2,i)
                imagesc(G{i});
            end
        end
        % create a model and data for each G
        data_rm = cell(1,size(G,2));
        C = data_rm;
        G_data = C;
        mu = zeros(1,size(G,2));
        RDM = zeros(size(G,2),condN*(condN-1)/2);
        for i=1:size(G,2)
            M.Ac = G{i};
            M.numGparams = 1;
            M.theta      = 1;
            M.type       = 'feature';
            
            S.numPart = nPart;
            S.numVox  = nVox;
            [data,~,~] = pcm_generateData(M,theta,S,1,1,0);
            % remove the mean from the data
            data_rm{i} = bsxfun(@minus,data{1},mean(data{1},1));
            C{i} = cov(data_rm{i}');
            G_data{i} = data_rm{i}*data_rm{i}';
            mu{i} = mean(data_rm{i},2);
            RDM(i,:)=pdist(data_rm{i});
        end
        
        % initialize distance for KL divergence
        D = zeros(size(G,2));
        cosDist = zeros(size(G));
        distCorr = zeros(size(G));
        for j=1:size(G,2)
            for k=1:size(G,2)
                D(j,k)=KLdivergence(C{j},C{k},mu{j},mu{k});
                cosDist(j,k)=pdist(RDM([j,k],:),'cosine');
                distCorr(j,k)=rsa_calcDistCorrRDMs(RDM([j,k],:));
            end
        end
        varargout{1}=distCorr;
    case 'dist_cov' % test case for distance covariances
        nCond = 5;
        nPart = 8;
        nVox = 100;
        numSim = 50;
        varReg = 1;
        corrReg = 0;
        type=3;
        vararginoptions(varargin,{'nCond','numSim','RDMtype','corrRDM','type','noiseType','corrReg','varReg'});
        
        [G,D] = makeGs(nCond,type);
        %  prepare model
        nRDM = size(G,2);
        trueRDM = zeros(nRDM,nCond*(nCond-1)/2);
        M = cell(1,nRDM);
        for i=1:nRDM
            trueRDM(i,:)=rsa_vectorizeRDM(D{i});
            M{i}=makeModel('sameRDM',G{i},nCond);
        end
        % other details for data generation
        S.numPart = nPart;
        S.numVox  = nVox;
        % calculate true distances
        [T.trueDist,T.reg1,T.reg2,T.distType] = calcTrueDist(trueRDM);
        NN=[];
        data=cell(1,nRDM);
        for cor=corrReg
            for var=varReg
                for n=1:numSim
                    for i=1:nRDM
                        [data{i},partVec,condVec] = makePatterns(M{i}.Gc,'signal',1,'nPart',S.numPart,'nVox',S.numVox); %signal 1, noise 0
                    end
                    % add shared noise across regions
                    data = addSharedNoise(data,var,cor);
                    [V,~,~]=covariance_dist(partVec,condVec,'G',M{i}.Gc,'sigma',eye(nCond)*var,'nVox',nVox);
                    V = pcm_makePD(V);
                    dist = rsa.distanceLDC(data{4},partVec,condVec);
                    dist_pw = rdm_prewhitenDist(dist,V);
                    % store info
                    N.dist      = dist';
                    N.dist_pw   = dist_pw';
                    N.distID    = (1:size(dist,2))';
                    N.numSim    = ones(size(N.dist))*n;
                    N.varReg    = ones(size(N.dist))*var;
                    N.corrReg   = ones(size(N.dist))*cor;
                    NN=addstruct(NN,N);
                end; % number of simulations
            end
        end
        
        % different variance of within-reg noise
        T=getrow(NN,NN.corrReg==0);
        figure
        scatterplot(T.dist,T.dist_pw,'split',T.varReg);
        xlabel('distance');
        ylabel('distance-pw');
        i1=pivottable(T.varReg,T.distID,T.dist,'mean');
        i2=pivottable(T.varReg,T.distID,T.dist_pw,'mean');
        figure
        vlen=length(varReg);
        for v=1:vlen
            subplot(vlen,2,(v-1)*2+1)
            imagesc(rsa_squareRDM(i1(v,:)));
            title(sprintf('distance var %d',varReg(v)));
            subplot(vlen,2,(v-1)*2+2)
            imagesc(rsa_squareRDM(i2(v,:)));
            title(sprintf('distance-pw var %d',varReg(v)));
            hold on;
        end
        % different variance of shared noise
        T=getrow(NN,NN.varReg==20);
        figure
        scatterplot(NN.dist,NN.dist_pw,'split',NN.corrReg);
        xlabel('distance');
        ylabel('distance-pw');
        i1=pivottable(NN.corrReg,NN.distID,NN.dist,'mean');
        i2=pivottable(NN.corrReg,NN.distID,NN.dist_pw,'mean');
        figure
        vlen=length(varReg);
        for v=1:vlen
            subplot(vlen,2,(v-1)*2+1)
            imagesc(rsa_squareRDM(i1(v,:)));
            title(sprintf('distance corr %d',corrReg(v)));
            subplot(vlen,2,(v-1)*2+2)
            imagesc(rsa_squareRDM(i2(v,:)));
            title(sprintf('distance-pw corr %d',corrReg(v)));
            hold on;
        end
    case 'dist_cov_level1'
        nCond = 4;
        distFactor = 0.5:0.01:1.5; % constant to multiply the distance with
        %noiseL = [0.01,0.1,1,5,20];
        noiseL = 0.1;
        nVox=100;
        nPart = 8;
        numSim=1000;
        runEmpirical = 0; % runEmpirical simulation or not
        vararginoptions(varargin,{'nVox','nCond','nPart','noiseL','numSim','runEmpirical'});
        % create G / D
        U = normrnd(0,1,[4,6]);
        G = U*U';
        G = G./trace(G); % normalize G
        C = indicatorMatrix('allpairs',unique(1:nCond)); % contrast matrix
        D = rsa_squareRDM(diag(C*G*C')');% true distance matrix - initial
        H = eye(nCond) - 1/nCond;
        
        TT=[];
        for v=noiseL
            for f=distFactor
                dist=zeros(numSim,size(C,1));
                D(1,2) = D(1,2)*f;    % alter 1 distance
                D(2,1) = D(1,2);
                G = -0.5*H*D*H'; % new G
                if runEmpirical
                    for i=1:numSim
                        [data{1},partVec,condVec] = makePatterns(G,'signal',1,'nPart',nPart,'nVox',nVox); %signal 1, noise 0
                        data = addSharedNoise(data,v,0,nCond);
                        % recalculate distances
                        dist(i,:) = makeRDM(data,partVec,condVec,'crossnobis');
                    end
                    % estimate covariance of distances - empirically
                    V_em = cov(dist);
                    T.V_em = rsa_vectorizeIPMfull(V_em);
                else
                    partVec = kron((1:nPart)',ones(nCond,1));            % Partitions
                    condVec = kron(ones(nPart,1),(1:nCond)');
                end
                % calculate covariance of distances - theoretically
                [V_th,delta,ksi]=covariance_dist(partVec,condVec,'G',G,'sigma',eye(nCond)*v,'nVox',nVox);
                T.dist = rsa_vectorizeRDM(D);
                T.V_th = rsa_vectorizeIPMfull(V_th);
                T.delta = rsa_vectorizeIPMfull(delta);
                T.ksi = rsa_vectorizeIPMfull(ksi);
                T.noiseLevel = v;
                TT=addstruct(TT,T);
            end
            keyboard;
            figure
            plot(TT.dist(:,1),TT.V_th(:,1),'o-');
            xlabel('Distance d12')
            title('variance / covariance');
            hold on;
            plot(TT.dist(:,1),TT.V_th(:,5),'ro-');
            hold on;
            % plot(TT.dist(:,1),TT.V_th(:,2),'ro-');
            plot(TT.dist(:,1),TT.V_th(:,end),'go-');
        end
        
    case 'run_job'
        %rsa_connect('run_simulation');
        typeNum = [11,12,13,21,22,23,3];
        for t=typeNum
            % rsa_connect('run_simulation','noiseType','within','type',t,'noiseType','between');
            rsa_connect('run_simulation','type',t,'noiseType','between');
            fprintf('Done type %d\n',t);
        end
        
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions

function [G,D]  = makeGs(condN,type)
% makes specific Gs
switch type
    case 11 % a1 < a2 < a3 (same, diff, orthogonal)
        D{1}=zeros(condN); D{2}=D{1};
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:condN+1:end)=0; %
        D{4}(3:5,1:5)=1;
        D{4}(1:5,3:5)=1;
        D{4}(1:condN+1:end)=0;
        D{2}=D{1};
        D{3}=0.7*D{1}+0.3*D{4};
    case 12
        D{1}=zeros(condN); D{2}=D{1};
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:condN+1:end)=0; %
        D{4}(3:5,1:5)=1;
        D{4}(1:5,3:5)=1;
        D{4}(1:condN+1:end)=0;
        D{2}=D{1};
        D{3}=0.5*D{1}+0.5*D{4};
    case 13
        D{1}=zeros(condN); D{2}=D{1};
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:condN+1:end)=0; %
        D{4}(3:5,1:5)=1;
        D{4}(1:5,3:5)=1;
        D{4}(1:condN+1:end)=0;
        D{2}=D{1};
        D{3}=0.3*D{1}+0.7*D{4};
    case 21 % a1 < a2 < a3 (same, diff, more diff)
        D{1}=zeros(condN); 
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:condN+1:end)=0; %
        tmp(3:5,3:5)=1;
        tmp(1:condN+1:end)=0;
        tmp(1:condN+1:end)=0;
        D{2}=D{1};
        U3 = normrnd(0,1,[5,6]);
        D3 = U3*U3';
        D3(1:5+1:end)=0;
        D{3}=0.2*D3+0.9*(D{1}*0.9+tmp*0.1);
        D{4}=0.2*D3+0.5*(D{1}*0.9+tmp*0.1);
    case 22 % a1 < a2 < a3 (same, diff, more diff)
        D{1}=zeros(condN); 
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:condN+1:end)=0; %
        tmp(3:5,3:5)=1;
        tmp(1:condN+1:end)=0;
        tmp(1:condN+1:end)=0;
       % D{2}=D{1};
        U3 = normrnd(0,1,[5,6]);
        D3 = U3*U3';
        D3(1:5+1:end)=0;
        D{2}=0.2*D3+0.9*(D{1}*0.9+tmp*0.1);
        D{3}=0.2*D3+0.7*(D{1}*0.9+tmp*0.1);
        D{4}=0.2*D3+0.5*(D{1}*0.9+tmp*0.1);
    case 23
        U = normrnd(0,1,5,6);
        D{1}=U*U';
        D{1}(1:condN+1:end)=0; %
        %tmp(3:5,3:5)=1;
       % D{2}=D{1};
        U3 = normrnd(0,1,[5,6]);
        D3 = U3*U3';
        D3(1:5+1:end)=0;
        D{2}=0.8*D{1}+0.2*D3; 
        D{3}=0.5*D{1}+0.5*D3; 
        D{4}=0.2*D{1}+0.8*D3; 
    case 3 % special case rank def D4
        D{1}=ones(condN)-eye(condN);
        D{2}=D{1};
        D{4}=zeros(condN);
        D{4}(1,1:5)=1;
        D{4}(2:5,1)=1;
        D{4}(1:condN+1:end)=0; %
        D{3}=0.5*D{1}+0.5*D{4};
end
H = eye(condN) - 1/condN; 
H1 = indicatorMatrix('allpairs',(1:condN));  
G = cell(1,4);
for i=1:4
    G{i} = -0.5*H*D{i}*H';
    G{i} = pcm_makePD(G{i});
    G{i} = G{i}./trace(G{i});
    G{i}(isnan(G{i}))=0; % in case of nans
    % recalc D
    D{i}=rsa_squareRDM(diag(H1*G{i}*H1')');
end
        
end
function M      = makeModel(rdmType,D,nCond)
M.type       = 'feature';
%M.numGparams = 1;
M.numGparams = 1;
M.theta      = 1; 
switch rdmType
    case 'sameRDM'
        M.Gc=D;
    case 'randomRDM'
        M.Ac=rand(nCond);
        M.Gc = M.Ac*M.Ac';
    case 'combRDM'
        M.Gc=D;
end
end
function data   = addSharedNoise(data,var,r)
% input: 
% data - datasets
% alpha - variance
% r - correlation
    nDataset = size(data,2);
    nVox    = size(data{1},2);
    nTrials = size(data{1},1);
    Z = normrnd(0,1,nTrials,nDataset*nVox);
    A = eye(nDataset)*var+(ones(nDataset)-eye(nDataset))*var*r;
    P = kron(A,eye(nVox)); 
    Zn = Z*sqrtm(P);     % shared noise matrix across reg
    for i=1:nDataset
        data{i} = data{i} + Zn(:,(i-1)*nVox+1:i*nVox);
    end
end

function [trueDist,ind1,ind2,distType]  = calcTrueDist(trueRDM)

[trueDist1,~,~] = calcDist(trueRDM,'correlation');  % 1 - squared + correlation
[trueDist2,~,~] = calcDist(trueRDM,'cosine');       % 2 - squared + cosine
tmpRDM = ssqrt(trueRDM);
[trueDist3,~,~] = calcDist(tmpRDM,'cosine');        % 3 - ssqrt + cosine
V = repmat({eye(size(trueRDM,2))},1,size(trueRDM,1));
% 'prewhiten' distances
tmpRDM = rdm_prewhitenDist(trueRDM,V,'type','uni');
[trueDist4,~,~] = calcDist(tmpRDM,'cosine');  % 4 - prewhiten + cosine (same as squared here)
tmpRDM = rdm_prewhitenDist(trueRDM,V,'type','multi');
[trueDist5,reg1,reg2] = calcDist(tmpRDM,'cosine');  % 4 - prewhiten + cosine (same as squared here)

% concatenate all results
trueDist = [trueDist1; trueDist2; trueDist3; trueDist4; trueDist5];
ind1 = repmat(reg1,4,1);
ind2 = repmat(reg2,4,1);
distType = kron((1:4)',ones(size(trueDist4,1),1));
end
function [dist,ind1,ind2,distType]      = calcDistAll(data,partVec,condVec,distType,V)

switch distType
    case 1 % correlation non-cv
        rdm = makeRDM(data,partVec,condVec,'correlation'); % previously sqEuc
        [dist,ind1,ind2] = calcDist(rdm,'correlation');
    case 2 % cosine single cv crossnobis (squared)
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 3 % cosine single cv crossnobis (ssqrt)
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        rdm = ssqrt(rdm);
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 4 % cosine single cv crossnobis (covariance uni-prewhitened)
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        rdm = rdm_prewhitenDist(rdm,V,'type','uni');
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 5 % cosine single cv crossnobis (covariance multi-prewhitened)
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        rdm = rdm_prewhitenDist(rdm,V);
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 6 % anzellotti
        [dist,ind1,ind2] = anzellottiDist(data,partVec,condVec);
end
distType = repmat(distType,size(dist,1),1);
end
function [dist,ind1,ind2,distType]      = calcTrueDistAll(trueRDM,distType,V)
switch distType
    case 1 % correlation non-cv
        [dist,ind1,ind2] = calcDist(trueRDM,'correlation');
    case 2 % cosine single cv crossnobis (squared)
        [dist,ind1,ind2] = calcDist(trueRDM,'cosine');
    case 3 % cosine single cv crossnobis (ssqrt)
        rdm = ssqrt(trueRDM);
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 4 % cosine single cv crossnobis (covariance uni-prewhitened)
        rdm = rdm_prewhitenDist(trueRDM,V,'type','uni');
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 5 % cosine single cv crossnobis (covariance multi-prewhitened)
        rdm = rdm_prewhitenDist(trueRDM,V);
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 6 % anzellotti
        dist = [0;0;0];
        ind1 = [1;1;1];
        ind2 = [2;3;4];
end
distType = repmat(distType,3,1);
end


function rdm   = makeRDM(data,partVec,condVec,type)
% function rdm   = makeRDM(data,partVec,condVec)
% makes RDM matrix with given type
nData=size(data,2);
X = indicatorMatrix('identity_p',condVec);
condN = length(unique(condVec));
H=indicatorMatrix('allpairs',unique(condVec)');
for st=1:nData
    nVox = size(data{st},2); 
    D = cell(1,nData);
    rdm = zeros(nData,condN*(condN-1)/2);
    switch type
        case 'sqEuc' % squared euclidean
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            G=D{st}*D{st}';
            rdm(st,:)= diag(H*G*H')/nVox;
        case 'crossnobis' % Squared crossvalidated Euclidean (crossnobis)
            % calculate crossvalidated squared Euclidean distances
            rdm(st,:)=rsa.distanceLDC(data{st},partVec,condVec);
        case 'correlation' % correlation
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            D{st}=bsxfun(@minus,D{st},mean(D{st},1));
            %G=D{st}*D{st}';
            % calculate correlation distance (mean subtracted)
            %rdm(st,:)=corr_crossval(G,'reg','minvalue');
            rdm(st,:)=rsa_vectorizeRDM(1-corr(D{st}'));
        case 'cosine' % cosine
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            G=D{st}*D{st}';
            % calculate cosine distance (mean not subtracted)
            rdm(st,:)=corr_crossval(G,'reg','minvalue');
    end
end;

end
function [dist,ind1,ind2] = calcDist(rdm,distType)
% calculate distance metric from the input
% N datasets, D distances
% input: N x D matrix
% dist types:
% 1) correlation
% 2) cosine
% 3) euclidean
% output: dist - distances
% ind1: indicator which rdm taken as first
% ind2: indicator which rdm taken as second
switch (distType)
    case 'correlation'
        % additional step for correlation - first remove the mean
        rdm  = bsxfun(@minus,rdm,mean(rdm,2));
end
nRDM = size(rdm,1);
rdm  = normalizeX(rdm);
tmpR  = rdm*rdm';
%distAll = 1-rsa_vectorizeRDM(tmpR);
dist = 1-rsa_vectorizeRDM(tmpR)';
%distAll = 1-tmp(:,1);
%dist = distAll(1+1:size(rdm,1));
ind=indicatorMatrix('allpairs',(1:nRDM));
ind1=zeros(size(ind,1),1);
ind2=zeros(size(ind,1),1);
for i=1:size(ind,1)
    j=find(ind(i,:));
    ind1(i,:)=j(1);
    ind2(i,:)=j(2);
end
end

function [dist,ind1,ind2] = anzellottiDist(data,partVec,condVec)
% function [dist,ind1,ind2,distType] = anzellottiDist(data,partVec,condVec)
% calculates a relationship between data of different regions
% for now the distance output is 1-R2 and 1-r
ind1 = ones(size(data,2)-1,1);
ind2 = ind1;
dist = zeros(size(ind1,1),1);
for i=1:size(ind1,1)
    dist(i) = multiDependVox(data{i+1},data{1},partVec,condVec,'type','reduceAB');
    ind2(i,:)=i+1;
end
end
