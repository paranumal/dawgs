
ids = find(([data.("Nvectors")]' == 0) & ([data.("Degree")]' == 3));
sub_data= data(ids);

ids = find([sub_data.("Exchange")]'==1);
AR_data= sub_data(ids);

ids = find([sub_data.("Exchange")]'==2);
PW_data= sub_data(ids);

ids = find([sub_data.("Exchange")]'==3);
CR_data= sub_data(ids);

%{
Nranks = [2, 4, 6, 12, 18, 24, 36, 48];
labels = {"N1n2"; "N1n4"; "N1n6";
          "N2n12"; "N3n18"; "N4n24"; "N6n36"; "N8n48"};
%}

Nranks = [2, 4, 8, 12, 16, 20, 24, 28];
labels = {"N1n2"; "N1n4"; "N2n8";
          "N3n12"; "N4n16"; "N5n20"; "N6n24"; "N7n28"};

figure(1)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == true) & ([AR_data.("GPU-aware")]' == false));
  AR_data_N= AR_data(ids);
  
  NLdofs = [AR_data_N.("Local_Dofs")]';
  dprs = [AR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("AR Exchange");
legend ("location", "northwest");
hold off


figure(2)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == true) & ([PW_data.("GPU-aware")]' == false));
  PW_data_N= PW_data(ids);
  
  NLdofs = [PW_data_N.("Local_Dofs")]';
  dprs = [PW_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("PW Exchange");
legend ("location", "northwest");
hold off


figure(3)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == true) & ([CR_data.("GPU-aware")]' == false));
  CR_data_N= CR_data(ids);
  
  NLdofs = [CR_data_N.("Local_Dofs")]';
  dprs = [CR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("CR Exchange");
legend ("location", "northwest");
hold off

%{
figure(4)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == false) & ([AR_data.("GPU-aware")]' == true));
  AR_data_N= AR_data(ids);
  
  NLdofs = [AR_data_N.("Local_Dofs")]';
  dprs = [AR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("AR Exchange GPU-Aware");
legend ("location", "northwest");
hold off


figure(5)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == false) & ([PW_data.("GPU-aware")]' == true));
  PW_data_N= PW_data(ids);
  
  NLdofs = [PW_data_N.("Local_Dofs")]';
  dprs = [PW_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("PW Exchange GPU-Aware");
legend ("location", "northwest");
hold off


figure(6)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == false) & ([CR_data.("GPU-aware")]' == true));
  CR_data_N= CR_data(ids);
  
  NLdofs = [CR_data_N.("Local_Dofs")]';
  dprs = [CR_data_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("CR Exchange GPU-Aware");
legend ("location", "northwest");
hold off
%}

figure(7)
clf
hold on
for N=1:size(Nranks,2)
  ids = find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == false) & ([CR_data.("GPU-aware")]' == false));
  data_N= CR_data(ids);
  
  NLdofs = [data_N.("Local_Dofs")]';
  dprs = [data_N.("Dofs/rank*s")]';
  
  %data_N = CR_data(find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == false) & ([CR_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  data_N = CR_data(find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == true) & ([CR_data.("GPU-aware")]' == false)));
  dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  %data_N = CR_data(find(([CR_data.("Nranks")]'==Nranks(N)) & ([CR_data.("Overlap")]' == true) & ([CR_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  
  data_N = AR_data(find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == false) & ([AR_data.("GPU-aware")]' == false)));
  dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  %data_N = AR_data(find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == false) & ([AR_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  data_N = AR_data(find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == true) & ([AR_data.("GPU-aware")]' == false)));
  dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  %data_N = AR_data(find(([AR_data.("Nranks")]'==Nranks(N)) & ([AR_data.("Overlap")]' == true) & ([AR_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  
  data_N = PW_data(find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == false) & ([PW_data.("GPU-aware")]' == false)));
  dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  %data_N = PW_data(find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == false) & ([PW_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  data_N = PW_data(find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == true) & ([PW_data.("GPU-aware")]' == false)));
  dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  %data_N = PW_data(find(([PW_data.("Nranks")]'==Nranks(N)) & ([PW_data.("Overlap")]' == true) & ([PW_data.("GPU-aware")]' == true)));
  %dprs = max(dprs, [data_N.("Dofs/rank*s")]');
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("Best Exchange");
legend ("location", "northwest");
hold off


