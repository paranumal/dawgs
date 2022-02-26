%{%
Nranks = [2, 4, 6, 12, 18, 24, 36, 48];
labels = {"N1n2"; "N1n4"; "N1n6";
          "N2n12"; "N3n18"; "N4n24"; "N6n36"; "N8n48"};
%}

%{          
Nranks = [2, 4, 8, 12, 16, 20, 24, 28];
labels = {"N1n2"; "N1n4"; "N2n8"; "N3n12"; "N4n16"; "N5n20"; "N6n24"; "N7n28"};          
%}
          
figure(8)
clf
hold on
clear X Y   
for N=1:8
  ids = find(([refdata.("Nranks")]'==Nranks(N)));
  refdata_N= refdata(ids);
  
  NLdofs = [refdata_N.("Local_Dofs")]';
  dprs = [refdata_N.("Dofs/rank*s")]';
  
  X{N} = NLdofs';
  Y{N} = dprs';
  
  label = strcat(";", labels{N}, ";");
  semilogx(NLdofs, dprs, label);
endfor
title("gslib Exchange");
legend ("location", "northwest");
hold off

Legend = {"N1n2", "N1n4", "N1n6", "N2n12", "N3n18", "N4n24", "N6n36", "N8n48"};
%Legend = {"N1n2", "N1n4", "N2n8", "N3n12", "N4n16", "N5n20", "N6n24", "N7n28"};
Title = "Nvidia V100";
%Title = "AMD MI100";
XTitle = "Degrees of freedom per rank";
YTitle = "Throughput (DoFs per rank second)";
writeLatexFigure('V100_gslib.tex', X, Y, Title, XTitle, YTitle, Legend);
%writeLatexFigure('MI100_gslib.tex', X, Y, Title, XTitle, YTitle, Legend);