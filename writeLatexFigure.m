
function writeLatexFigure(filename, X, Y, Title, XTitle, YTitle, legend) 

fid = fopen (filename, "w");

fprintf(fid, "\\begin{tikzpicture}[scale=0.5]\n");

fprintf(fid, "\\begin{axis}[ \n \
  xmode=log, \n \
  %% ymode=log, \n \
  grid=both, \n \
  major grid style={line width=.1pt,draw=gray!50}, \n \
  minor grid style={line width=.1pt,draw=gray!50}, \n \
  domain=1:8, \n \
  width=4.5in, \n \
  ymin=1e-16, \n \
  xlabel={%s}, \n \
  ylabel={%s}, \n \
  cycle list name=will, \n \
  legend cell align=left, \n \
  legend pos=north west, \n \
  mark size=1.5pt, \n \
  line width=1.4pt, \n", XTitle, YTitle);
fprintf(fid, 'legend entries={');
for n=1:size(legend,2)-1
  fprintf(fid, "%s, ", legend{n});
endfor
fprintf(fid, "%s", legend{size(legend,2)});
fprintf(fid, "},\n");
fprintf(fid, "title={%s},\n", Title); 
fprintf(fid, "ymax=2.5e10,\n \
  ymin=0.0,\n \
  xmin=1e3,\n \
  xmax=1e8,\n \
]\n");
fprintf(fid, "\n");

for n=1:size(X,2)
  fprintf(fid, "\\addplot \
  table {%% %s \n", legend{n});
  for m=1:size(X{n},2)
    fprintf(fid, "%e %e \n", X{n}(m), Y{n}(m));
  endfor
  fprintf(fid, "};\n");  
  fprintf(fid, "\n");
endfor

fprintf(fid, "\\end{axis}\n");
fprintf(fid, "\\end{tikzpicture}\n");

fclose (fid);
end