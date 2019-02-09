function varargout = rsa_predictG(what,varargin)

switch what
    case 'transformG'
        U1=normrnd(0,1,5,6);
        U2 = U1;
        G1 = U1*U1';
        G2 = U2*U2';
        [U1,l1]=eig(G1);
        [U2,l2]=eig(G2);
        [l1,i1]=sort(diag(l1),1,'descend');
        [l2,i2]=sort(diag(l2),1,'descend');
        U1=U1(:,i1);
        U2=U2(:,i2);
        U1=bsxfun(@times,U1,sqrt(l1'));
        U2=bsxfun(@times,U2,sqrt(l2'));
        
        %A12 = pinv(U1)*U2;
        %A21 = pinv(U2)*U1;
        
        T12 = pinv(U1)*G2*pinv(U1)';
        T21 = pinv(U2)*G1*pinv(U2)';
        varargout{1}=T12;
        varargout{2}=T21;
    case 'mixG'
        % mix G3 as G1 and G2 with different proportions
        w1 = 0.1:0.1:0.9;
        
        U1=normrnd(0,1,[5,6]);
        U2=normrnd(0,1,[5,15]);
        G1 = U1*U1';
        G2 = U2*U2';
        G1 = G1./trace(G1);
        G2 = G2./trace(G2);
        
        for f=1:length(w1)
            G3=w1(f)*G1 + (1-w1(f))*G2;
            % decompose G1 - eigenvalue decomposition
            % G1
            [V1,L1]     = eig(G1);
            [l,i]       = sort(diag(L1),1,'descend'); % sort the eigenvalues
            V1          = V1(:,i);
            U1          = bsxfun(@times,V1,ssqrt(l'));
            % G2
            [V2,L2]     = eig(G2);
            [l,i]       = sort(diag(L2),1,'descend');
            V2          = V2(:,i);
            U2          = bsxfun(@times,V2,ssqrt(l'));
            % G3
            [V3,L3]     = eig(G3);
            [l,i]       = sort(diag(L3),1,'descend');
            V3          = V3(:,i);
            U3          = bsxfun(@times,V3,ssqrt(l'));
            
            % transformation matrix A and T - A*A'
            A13 = pinv(U1)*U3;
            A23 = pinv(U2)*U3;
            T13 = A13*A13';
            T23 = A23*A23';
            
            % plot
            figure(f)
            subplot(241)
            imagesc(G1)
            title('G1')
            subplot(242)
            imagesc(G2)
            title('G2')
            subplot(243)
            imagesc(G3)
            title(sprintf('G3 = %1.1f x G1+%1.1f x G2',w1(f),1-w1(f)));
            subplot(245)
            imagesc(A13)
            title('A1-3');
            subplot(246)
            imagesc(A23)
            title('A2-3');
            subplot(247)
            imagesc(T13)
            title('T1-3');
            subplot(248)
            imagesc(T23)
            title('T2-3');
        end
    case 'featureMix'
        % predict a transformation based on features
        % features can be data-driven (eigenvectors) or pre-specified
        
    case 'rank-def'
        % examine case when G1 rank deficient, G2 not
        U1=normrnd(0,1,[5,3]);
        U2=normrnd(0,1,[5,6]);
        G1 = U1*U1';
        G2 = U2*U2';
        [U1,l1]=eig(G1);
        l1=round(l1,3);
        [U2,l2]=eig(G2);
        [l1,i1]=sort(diag(l1),1,'descend');
        [l2,i2]=sort(diag(l2),1,'descend');
        U1=U1(:,i1);
        U2=U2(:,i2);
        U1=bsxfun(@times,U1,sqrt(l1'));
        U2=bsxfun(@times,U2,sqrt(l2'));
        
        A12 = pinv(U1)*U2;
        A21 = pinv(U2)*U1;
        
        T12 = pinv(U1)*G2*pinv(U1)';
        T21 = pinv(U2)*G1*pinv(U2)';
        
        predG2 = U1*T12*U1'; % cannot be recovered
        predG1 = U2*T21*U2'; % can be
        
        figure
        subplot(431)
        imagesc(G1)
        title('G1')
        subplot(432)
        imagesc(G2)
        title('G2')
        subplot(434)
        imagesc(U1)
        title('U1')
        subplot(435)
        imagesc(U2)
        title('U2')
        subplot(437)
        imagesc(A12)
        title('A1-2')
        subplot(438)
        imagesc(T12)
        title('T1-2')
        subplot(439)
        imagesc(predG2)
        title('predicted G2 - not recovered')
        subplot(4,3,10)
        imagesc(A21)
        title('A2-1')
        subplot(4,3,11)
        imagesc(T21)
        title('T2-1')
        subplot(4,3,12)
        imagesc(predG1)
        title('predicted G1 - recovered fully')
    case 'scaling'
        U1=normrnd(0,1,[5,6]);
        U2 = U1;
        G1 = U1*U1';
        G2 = U2*U2';
        [U1,l1]=eig(G1);
        [U2,l2]=eig(G2);
        [l1,i1]=sort(diag(l1),1,'descend');
        [l2,i2]=sort(diag(l2),1,'descend');
        U1=U1(:,i1);
        U2=U2(:,i2);
        U1=bsxfun(@times,U1,sqrt(l1'));
        U2=bsxfun(@times,U2,sqrt(l2'));
        
        %A12 = pinv(U1)*U2;
        %A21 = pinv(U2)*U1;
        
        T12 = pinv(U1)*G2*pinv(U1)';
        T21 = pinv(U2)*G1*pinv(U2)';
        
        predG2 = U1*T12*U1'; 
        predG1 = U2*T21*U2'; 
        
        % determine if pure scaling
        T12=round(T12);
        if isdiag(T12)
            scalDim = unique(diag(T12));
            if length(scalDim)==1
                scaling = 'isotropic';
            else
                scaling = 'non-isotropic';
            end
        end
        varargout{1}=scaling;
        varargout{2}=predG2;
        varargout{3}=predG1;
    case 'distanceMetric'
        % determine a distance metric from predG to realG
        
    case 'exampleFeature'
        % first make G1
        U1=normrnd(0,1,[5,6]);
        G1 = U1*U1';
        [U1,l1]=eig(G1);
        [l1,i1]=sort(diag(l1),1,'descend');
        U1=U1(:,i1);
        U1=bsxfun(@times,U1,sqrt(l1'));
        % exaggerate the first feature in G2 3x in U
        U2=bsxfun(@times,U1,[3,1,1,1,1]);
        G2 = U2*U2';
        
        % determine weights of features based on feature set U1
        w1=pinv(U1)*G1*pinv(U1)';
        w1=round(w1,3);
        if isdiag(w1)
            w2=pinv(U1)*G2*pinv(U1)'; % feature set from G1
            w2=round(w2,3);
            figure
            subplot(331)
            imagesc(G1);
            title('G1');
            subplot(332)
            imagesc(G2);
            title('G2');
            subplot(333)
            imagesc(U1);
            title('Feature set F1 (from U1)');
            subplot(334)
            imagesc(w1)
            title('Weights for G1');
            subplot(335)
            imagesc(w2)
            title('Weights for G2');
            subplot(337)
            FT = w2./w1;
            FT(isnan(FT))=0;
            imagesc(FT);
            title('FT - w2./w1');
            subplot(338)
            scatterplot(diag(w1),diag(w2),'label',(1:5));
            min_axis = min([range(xlim) range(ylim)]);
            hold on;
            plot(0:min_axis,0:min_axis,'k-');
            xlabel('w1');
            ylabel('w2');
            title('FT');
            subplot(339)
            T=pinv(U1)*G2*pinv(U1)';
            imagesc(T)
            title('T - G1->G2');
        end
        %G_pred = U1*T*U1';

        
        % now make a new G1 (leaving U as is)
        U1_new=bsxfun(@times,U1,[2 1.5 1.3 1 0.5]);
        G1_new=U1_new*U1_new';
        U2_new=bsxfun(@times,U1_new,[3 1 1 1 1]);
        G2_new=U2_new*U2_new';
        w1=pinv(U1)*G1_new*pinv(U1)';
        w1=round(w1,3);
        if isdiag(w1)
            w2=pinv(U1)*G2_new*pinv(U1)';
            w2=round(w2,3);
            FT = w2./w1;
            FT(isnan(FT))=0;
            figure
            subplot(341)
            imagesc(G1_new);
            title('G1');
            subplot(342)
            imagesc(G2_new);
            title('G2');
            subplot(343)
            imagesc(U1);
            title('Feature set F1');
            subplot(344)
            imagesc(U1_new);
            title('Real feature set in G1 (from U1)');
            subplot(345)
            imagesc(w1)
            title('Weights for G1');
            subplot(346)
            imagesc(w2)
            title('Weights for G2');
            subplot(349)
            imagesc(FT);
            title('FT - w2./w1');
            subplot(3,4,10)
            scatterplot(diag(w1),diag(w2),'label',(1:5));
            min_axis = min([range(xlim) range(ylim)]);
            hold on;
            plot(0:min_axis,0:min_axis,'k-');
            xlabel('w1');
            ylabel('w2');
            title('T');
            subplot(3,4,11)
            T=pinv(U1_new)*G2_new*pinv(U1_new)';
            imagesc(T)
            title('T - G1->G2');
        end
        %G_pred = U1_new*T*U1_new';
        %equivalent to: G_pred = U1_new*FT*U1_new';
        
        % now make a different G (w not diagonal)
        U3=normrnd(0,1,[5,10]);
        G3 = U3*U3';
        [U3,l3]=eig(G3);
        [l3,~]=sort(diag(l3),1,'descend');
        U3=U3(:,i1);
        U3=bsxfun(@times,U3,sqrt(l3')); 
        % exaggerate the first feature in G4 3x in U
        U4=bsxfun(@times,U3,[3,1,1,1,1]);
        G4 = U4*U4';
        
        w3 = pinv(U1)*G3*pinv(U1)';
        % calculate transformation from G3 to G4
        A = pinv(U3)*U4;
        w4 = A*w3*A';
        FT = w4./w3;
        FT(isnan(FT))=0;
        figure
        subplot(341)
        imagesc(G3);
        title('G1');
        subplot(342)
        imagesc(G4);
        title('G2');
        subplot(343)
        imagesc(U1);
        title('Feature set F1');
        subplot(344)
        imagesc(U3);
        title('Real feature set in G1 (from U1)');
        subplot(345)
        imagesc(w3);
        title('Weights for G1');
        subplot(346)
        imagesc(w4);
        title('Weights for G2');
        subplot(349)
        imagesc(FT);
        title('FT - w2./w1');
        subplot(3,4,10)
        scatterplot(diag(w3),diag(w4),'label',(1:5));
        min_axis = min([range(xlim) range(ylim)]);
        hold on;
        plot(0:min_axis,0:min_axis,'k-');
        xlabel('w1');
        ylabel('w2');
        title('FT');
        subplot(3,4,11)
        T=pinv(U3)*G4*pinv(U3)';
        imagesc(T);
        title('T - G1->G2');
        
        
        
    otherwise
        fprintf('No such case!')
end