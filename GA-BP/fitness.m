function error = fitness(x)

load data inputnum H1_best H2_best H3_best outputnum net inputn outputn output_train inputn_test outputps output_test

setdemorandstream(pi);

w1=x(1:inputnum*H1_best);
B1=x(inputnum*H1_best+1:inputnum*H1_best+H1_best);

w2=x(inputnum*H1_best+H1_best+1 : inputnum*H1_best+H1_best+H1_best*H2_best);
B2=x(inputnum*H1_best+H1_best+H1_best*H2_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best);


w3=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best);
B3=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best);

w4=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum);
B4=x(inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum+1:inputnum*H1_best+H1_best+H1_best*H2_best+H2_best+H2_best*H3_best+H3_best+H3_best*outputnum+outputnum);

net.trainParam.showWindow=0;  

net.iw{1,1}=reshape(w1,H1_best,inputnum);
net.lw{2,1}=reshape(w2,H2_best,H1_best);
net.lw{3,2}=reshape(w3,H3_best,H2_best);
net.lw{4,3}=reshape(w4,outputnum,H3_best);

net.b{1}=reshape(B1,H1_best,1);
net.b{2}=reshape(B2,H2_best,1);
net.b{3}=reshape(B3,H3_best,1);
net.b{4}=reshape(B4,outputnum,1);

net=train(net,inputn,outputn);

an0=sim(net,inputn);
train_simu=mapminmax('reverse',an0,outputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
 
error=(mse(output_train,train_simu)+mse(output_test,test_simu))/2; 



