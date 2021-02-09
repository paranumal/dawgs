
Nranks = [2, 4, 6, 12, 18, 24, 36, 48];
labels = {"N1n2"; "N1n4"; "N1n6";
          "N2n12"; "N3n18"; "N4n24"; "N6n36"; "N8n48"};

figure(8)
clf
hold on
for N=1:8
  ids = find(([refdata.("Nranks")]'==Nranks(N)));
  refdata_N= refdata(ids);
  
  NLdofs = [refdata_N.("Local_Dofs")]';
  dprs = [refdata_N.("Dofs/rank*s")]';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("gslib Exchange");
legend ("location", "northwest");
hold off

