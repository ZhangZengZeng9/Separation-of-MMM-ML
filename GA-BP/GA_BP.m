%%  Save the result to be displayed on the command line to 'D:\mat_file\123.txt '
diary('D:\mat_file\123.txt'); 

%% Initialize
clear
close all
clc

%% Read data
all=xlsread('ML.xlsx','A2:O288'); % Read the data of the corresponding range in excel
rowrank = randperm(size(all, 1)); 
D=all(rowrank,:);
trainx=D(:,1:14);
trainy=D(:,15:15);
p_train=trainx;
t_train=trainy;
arr = [ 'logsig' ,'tansig' ];

%% Obtain the neural network structure and results
inputnum=size(trainx,2);
outputnum=size(trainy,2);
diary on;
disp('/////////////////////////////////')
disp('ANN structure...')
disp(['The number of nodes in the input layer is：',num2str(inputnum)])
disp(['The number of nodes in the output layer is：',num2str(outputnum)])
disp(' ')
disp('The process of determining hidden layer nodes...')
diary off;

%% Determine the number of hidden layer neurons
MSE=1e+5; 
R2=0;
for H1=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10   
    for H2=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
        for H3=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
             cv=5;
            indices = crossvalind('Kfold',length(p_train),cv);
            for H1_tf1 = 1:2
                if H1_tf1 ==1
                   tf1 = 'tansig';
                elseif H1_tf1 ==2
                   tf1 = 'logsig' ;
                end
                for H2_tf2 = 1:2
                    if H2_tf2 ==1
                       tf2 = 'tansig';
                    elseif H2_tf2 ==2
                       tf2 = 'logsig' ;
                    end
                    for H3_tf3 = 1:2
                        if H3_tf3 ==1
                           tf3 = 'tansig';
                        elseif H3_tf3 ==2
                           tf3 = 'logsig' ;
                        end 
                    diary on; 
                    disp(['The number of hidden layer nodes is ',num2str(H1),' ',num2str(H2),' ',num2str(H3),' transfer function is ',num2str(tf1),' ',num2str(tf2),' ',num2str(tf3),])     
                    tempMse = 0;
                    tempR2 = 0;
                    for i = 1:cv
                        testa = (indices == i); traina = ~testa;
                        p_cv_train=p_train(traina,:);  
                        t_cv_train=t_train(traina,:); 
                        p_cv_test=p_train(testa,:); 
                        t_cv_test=t_train(testa,:); 
                        p_cv_train=p_cv_train';
                        t_cv_train=t_cv_train';
                        p_cv_test= p_cv_test';
                        t_cv_test= t_cv_test';
                        input_train = p_cv_train;
                        output_train =t_cv_train;
                        input_test = p_cv_test;
                        output_test =t_cv_test;
                        [inputn,inputps]=mapminmax(input_train,0,1);
                        [outputn,outputps]=mapminmax(output_train,0,1);
                        
                      %% Build a network
                        net=newff(inputn,outputn,[ H1 H2 H3 ],{ tf1,tf2,tf3 ,'purelin'},'trainlm');
                        net = init(net);
                        net.trainParam.epochs=10000;              
                        net.trainParam.lr=0.001;                   
                        net.trainParam.goal=0.0001;                
                        net.trainParam.showWindow=0;             
                        net=train(net,inputn,outputn);
                        inputn_test=mapminmax('apply',input_test,inputps,-1,1);
                        outputn_test=mapminmax('apply',output_test,outputps,-1,1);
                        an=sim(net,inputn_test); 
                        r2= regression(outputn_test,an);
                        mse0=mse(outputn_test,an);  
                        disp(['CV:',num2str(i),',MSE of test is：',num2str(mse0),'，Test set R2 is：',num2str(r2)])
                        tempMse = tempMse + mse0;
                        tempR2 = tempR2+r2;
                    end
                    disp(['mean MSE is ：',num2str(tempMse/cv),',mean R2 is：',num2str(tempR2/cv)]);
                    diary off; 
                    
                    %% Update the best hidden layer neuron structure
                    if (tempMse/cv)<MSE  
                        TF1_1=tf1;
                        TF2_1=tf2;
                        TF3_1=tf3;
                        MSE=tempMse/cv;
                        H1_best1=H1;
                        H2_best1=H2;
                        H3_best1=H3;
                        M_R2 = tempR2/cv;
                    end
                        if(tempR2/cv)>R2
                            TF1_2=tf1;
                            TF2_2=tf2;
                            TF3_2=tf3;
                            R2=tempR2/cv;
                            H1_best2=H1;
                            H2_best2=H2;
                            H3_best2=H3;
                            R_MSE=tempMse/cv;
                        end
                    end
                end
            end
        end
    end            
end
disp(' ')
diary on;
disp(['The optimal number of hidden layer nodes is：',num2str(H1_best1),' ',num2str(H2_best1),' ',num2str(H3_best1),',transfer function: ',num2str(TF1_1),' ',num2str(TF2_1),' ',num2str(TF3_1),',minimum mean MSE is：',num2str(MSE),',correspond average R2 is ：',num2str(M_R2)])
disp(['The optimal number of hidden layer nodes is：',num2str(H1_best2),' ',num2str(H2_best2),' ',num2str(H3_best2),',transfer function: ',num2str(TF1_2),' ',num2str(TF2_2),' ',num2str(TF3_2),',correspond average MSE is：',num2str(R_MSE),',maximum mean R2 is ：',num2str(R2)])

%% Data set partitioning
k=rand(1,291);
[m,n]=sort(k);
trainNum =232;
testNum = 291 - trainNum; 
input_train = all(n(1:trainNum),1:14)';
output_train = all(n(1:trainNum),15:15)';
input_test = all(n(trainNum+1:trainNum+testNum),1:14)';
output_test = all(n(trainNum+1:trainNum+testNum),15:15)';
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train,0,1);
H1_best=H1_best2;
H2_best=H2_best2;
H3_best=H3_best2;

%% BP neural network
disp(' ')
disp('BPANN：')
net0=newff(inputn,outputn,[H1_best H2_best H3_best],{TF1_2,TF2_2,TF3_2,'purelin'},'trainlm');
net0 = init(net0);

% Neural network parameter configuration
net0.trainParam.epochs=10000;                  
net0.trainParam.lr=0.001;                     
net0.trainParam.goal=0.0001;                  
net0.trainParam.show=25;                     
net0.trainParam.mc=0.01;                    
net0.trainParam.min_grad=1e-6;              
net0.trainParam.max_fail=6;                  

net0=train(net0,inputn,outputn);
inputn_test=mapminmax('apply',input_test,inputps,0,1);  
outputn_test=mapminmax('apply',output_test,outputps,0,1);  

an0=sim(net0,inputn_test); 
test_simu0=mapminmax('reverse',an0,outputps); 
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);

%% Genetic algorithm
trainNum = 232;
input_train = all(n(1:trainNum),1:14)';
output_train = all(n(1:trainNum),15:15)';
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train,0,1);
disp(' ')
disp('GABP-ANN：')
net = newff(inputn,outputn,[H1_best,H2_best,H3_best],{TF1_2,TF2_2,TF3_2,'purelin'},'trainlm');
net = init(net);
net.trainParam.epochs=10000;                  
net.trainParam.lr=0.01;                     
net.trainParam.goal=0.0001;                 
net.trainParam.show=25;                     
net.trainParam.mc=0.01;                   
net.trainParam.min_grad=1e-6;              
net.trainParam.max_fail=6;                 
save data inputnum H1_best H2_best H3_best outputnum net inputn outputn output_train  inputn_test outputps output_test

% Parameters of genetic algorithm 
PopulationSize_Data=50;               
MaxGenerations_Data=100;              
CrossoverFraction_Data=0.6;          
MigrationFraction_Data=0.01;       
nvars=inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum+outputnum;    %自变量个数
lb=repmat(-3,nvars,1);
ub=repmat(3,nvars,1);   

options = optimoptions('ga');
options = optimoptions(options,'PopulationSize', PopulationSize_Data);
options = optimoptions(options,'CrossoverFraction', CrossoverFraction_Data);
options = optimoptions(options,'MigrationFraction', MigrationFraction_Data);
options = optimoptions(options,'MaxGenerations', MaxGenerations_Data);
options = optimoptions(options,'SelectionFcn', @selectionroulette);  
options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);   
options = optimoptions(options,'MutationFcn', {  @mutationadaptfeasible [] [] });  
options = optimoptions(options,'Display', 'off');  
options = optimoptions(options,'PlotFcn', { @gaplotbestf });  

[x,fval] = ga(@fitness,nvars,[],[],[],[],lb,ub,[],[],options);
setdemorandstream(pi);
w1=x(1:inputnum*H1_best);
B1=x(inputnum*H1_best+1:inputnum*H1_best+H1_best);

w2=x(inputnum*H1_best+H1_best+1:inputnum*H1_best+H1_best+H1_best*H2_best);
B2=x(inputnum*H1_best+H1_best+H1_best*H2_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best);

w3=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best);
B3=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best);

w4=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum);
B4=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum+outputnum);

net.iw{1,1}=reshape(w1,H1_best,inputnum);
net.lw{2,1}=reshape(w2,H2_best,H1_best);
net.lw{3,2}=reshape(w3,H3_best,H2_best);
net.lw{4,3}=reshape(w4,outputnum,H3_best);

net.b{1}=reshape(B1,H1_best,1);
net.b{2}=reshape(B2,H2_best,1);
net.b{3}=reshape(B3,H3_best,1);
net.b{4}=reshape(B4,outputnum,1);

net=train(net,inputn,outputn);
an1=sim(net,inputn_test);
test_simu1=mapminmax('reverse',an1,outputps); 
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);
plotregression(outputn_test,an0,'BP Test',outputn_test,an1,'GA-BP Test')
save regression an0 an1 outputn_test

%% Draw
figure
plot(output_test,'kp-','markerfacecolor','k','linewidth',1.1)
hold on
plot(test_simu0,'bo-.','linewidth',1.1)
hold on
plot(test_simu1,'ro-.','markerfacecolor','r','linewidth',1.1)
legend('Target','BP Predict','GA-BP Predict','FontName','Times New Roman','FontWeight','bold','FontSize',16)
xlabel('Sample number','FontName','Times New Roman','FontWeight','bold','FontSize',200)
ylabel('Permeability','FontName','Times New Roman','FontWeight','bold','FontSize',200)
axis([0 32 -4 22]) 
set(gca,'XTick',(0:4:32),'linewidth',2) 
set(gca,'YTick',(-4:4:22),'linewidth',2)
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',15) 
set(gcf,'Position',[100 100 950 450])
grid on 

figure
plot(error0,'bo-.','linewidth',1.1)
hold on
plot(error1,'ro-.','markerfacecolor','r','linewidth',1.1)
hold on
legend('BP Error','GA-BP Error','FontName','Times New Roman','FontWeight','bold','FontSize',16)
xlabel('Sample number','FontName','Times New Roman','FontWeight','bold','FontSize',200)
ylabel('Error','FontName','Times New Roman','FontWeight','bold','FontSize',200)
axis([0 32 -8 8]) 
set(gca,'XTick',(0:4:32),'linewidth',2)
set(gca,'YTick',(-8:2:8),'linewidth',2) 
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',15) 
set(gcf,'Position',[100 100 950 450])
grid on
save duibitu output_test test_simu0 test_simu1 error0 error1

disp(' ')
disp('/////////////////////////////////')
disp('Print result table')
disp('Sample number  Target  BP-Predict  GABP-Predict  BP-Error  GABP-Error')
for i=1:testNum
    disp([i output_test(i), test_simu0(i), test_simu1(i), error0(i), error1(i)])
end
diary off;




