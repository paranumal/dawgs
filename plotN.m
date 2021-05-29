
ids = find(([data.("Nvectors")]' == 0) & ([data.("Overlap")]' == false) & ([data.("GPU-aware")]' == false)
            & ([data.("Nranks")]' == 4) );
sub_data= data(ids);

ids = find([sub_data.("Exchange")]'==1);
AR_data= sub_data(ids);

ids = find([sub_data.("Exchange")]'==2);
PW_data= sub_data(ids);

ids = find([sub_data.("Exchange")]'==3);
CR_data= sub_data(ids);


figure(1)
clf
hold on
for N=1:8
  ids = find([AR_data.("Degree")]'==N);
  AR_data_N= AR_data(ids);
  
  NLdofs = [AR_data_N.("Local_Dofs")]';
  dprs = [AR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";N = ", int2str(N), ";");
  semilogx(NLdofs, dprs, label);
endfor
title("AR Exchange");
legend ("location", "northwest");
hold off

figure(2)
clf
hold on
clear X Y     
for N=1:8
  ids = find([PW_data.("Degree")]'==N);
  PW_data_N= PW_data(ids);
  
  NLdofs = [PW_data_N.("Local_Dofs")]';
  dprs = [PW_data_N.("Dofs/rank*s")]';
  
  X{N} = NLdofs';
  Y{N} = dprs';
  
  label = strcat(";N = ", int2str(N), ";");
  semilogx(NLdofs, dprs, label);
endfor
title("PW Exchange");
legend ("location", "northwest");
hold off


Legend = {"$p=1$","$p=2$","$p=3$","$p=4$","$p=5$","$p=6$","$p=7$","$p=8$"};
%Title = "Nvidia V100";
Title = "AMD MI100";
XTitle = "Degrees of freedom per rank";
YTitle = "Throughput (DoFs per rank second)";
writeLatexFigure('MI100_PW_N.tex', X, Y, Title, XTitle, YTitle, Legend);

figure(3)
clf
hold on
for N=1:8
  ids = find([CR_data.("Degree")]'==N);
  CR_data_N= CR_data(ids);
  
  NLdofs = [CR_data_N.("Local_Dofs")]';
  dprs = [CR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";N = ", int2str(N), ";");
  semilogx(NLdofs, dprs, label);
endfor
title("CR Exchange");
legend ("location", "northwest");
hold off