
function writeLatexFigure(filename, X, Y, Title, XTitle, YTitle, legend) 

fid = fopen (filename, "w");

fprintf(fid, "\\begin{tikzpicture}[scale=0.75]\n");

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
fprintf(fid, "ymax=2.6e10\n \
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

%{
\addplot 
table {%p=2
216    2.359e+06
5832   5.9756e+07
27000   2.6002e+08
74088    6.826e+08
1.5746e+05   1.4049e+09
2.875e+05    2.513e+09
4.7455e+05   3.9274e+09
7.29e+05   5.9113e+09
1.0612e+06   8.3631e+09
1.4815e+06    1.045e+10
2.0004e+06   1.2097e+10
2.6281e+06   1.3398e+10
3.375e+06   1.4623e+10
4.2515e+06   1.5578e+10
5.268e+06    1.641e+10
6.4349e+06   1.7244e+10
7.7624e+06   1.7816e+10
9.261e+06   1.8016e+10
1.0941e+07   1.8227e+10
1.2813e+07   1.8175e+10
1.4887e+07   1.7844e+10
1.7174e+07   1.7543e+10
1.9683e+07   1.6962e+10
2.2426e+07   1.6388e+10
2.5412e+07   1.5648e+10
2.8653e+07   1.5312e+10
};

\addplot 
table {%p=3
512   5.5792e+06
13824   1.3458e+08
64000   5.9122e+08
1.7562e+05   1.5515e+09
3.7325e+05   3.1761e+09
6.8147e+05   5.4438e+09
1.1249e+06   8.7007e+09
1.728e+06   1.2334e+10
2.5155e+06   1.4647e+10
3.5118e+06    1.686e+10
4.7416e+06   1.8616e+10
6.2295e+06   2.0142e+10
8e+06   2.1234e+10
1.0078e+07   2.2129e+10
1.2487e+07   2.2854e+10
1.5253e+07   2.3414e+10
1.84e+07   2.3872e+10
2.1952e+07   2.4222e+10
2.5934e+07   2.4398e+10
3.0371e+07   2.4534e+10
3.5288e+07   2.4757e+10
};

\addplot 
table {%p=4
1000   1.0779e+07
27000    2.495e+08
1.25e+05   1.1054e+09
3.43e+05   2.9426e+09
7.29e+05   5.7465e+09
1.331e+06   9.9592e+09
2.197e+06   1.3285e+10
3.375e+06   1.5726e+10
4.913e+06   1.7569e+10
6.859e+06     1.88e+10
9.261e+06   1.9608e+10
1.2167e+07   1.9783e+10
1.5625e+07   1.9846e+10
1.9683e+07   1.9966e+10
2.4389e+07   2.0081e+10
2.9791e+07   2.0442e+10
};

\addplot 
table {%p=5
1728     1.81e+07
46656   4.3434e+08
2.16e+05   1.9005e+09
5.927e+05   4.7473e+09
1.2597e+06   9.4296e+09
2.3e+06   1.3863e+10
3.7964e+06   1.6827e+10
5.832e+06   1.9026e+10
8.4897e+06    2.053e+10
1.1852e+07   2.1259e+10
1.6003e+07    2.189e+10
2.1025e+07   2.2692e+10
2.7e+07   2.3088e+10
3.4012e+07   2.3342e+10
};

\addplot 
table {%p=6
2744   2.8624e+07
21952    2.018e+08
74088   6.7042e+08
1.7562e+05   1.5402e+09
3.43e+05    2.919e+09
5.927e+05   4.7353e+09
9.4119e+05   7.2115e+09
1.4049e+06   1.0048e+10
2.0004e+06     1.29e+10
2.744e+06    1.459e+10
3.6523e+06   1.5867e+10
4.7416e+06   1.7587e+10
6.0286e+06   1.8575e+10
7.5295e+06   1.9437e+10
9.261e+06   1.9937e+10
1.1239e+07   2.0368e+10
1.3481e+07   2.0650e+10
1.6003e+07   2.1033e+10
1.8821e+07   2.1338e+10

};

\addplot 
table {%p=7
4096   4.1698e+07
32768   3.0606e+08
1.1059e+05   9.7462e+08
2.6214e+05   2.2591e+09
5.12e+05   4.0884e+09
8.8474e+05   6.7928e+09
1.4049e+06   9.9672e+09
2.0972e+06   1.3361e+10
2.986e+06   1.5745e+10
4.096e+06   1.7538e+10
5.4518e+06   1.9107e+10
7.0779e+06   2.0302e+10
8.9989e+06    2.136e+10
1.1239e+07   2.2142e+10
};

\addplot 
table {%p=8
5832   5.7694e+07
46656   4.2587e+08
1.5746e+05   1.3804e+09
3.7325e+05   3.0934e+09
7.29e+05   5.6003e+09
1.2597e+06   8.9905e+09
2.0004e+06   1.2792e+10
2.986e+06   1.5148e+10
4.2515e+06   1.7011e+10
5.832e+06   1.8343e+10
7.7624e+06   1.9357e+10
1.0078e+07   2.0093e+10
1.2813e+07   2.0657e+10
1.6003e+07   2.1177e+10
};

%}


fprintf(fid, "\\end{axis}\n");
fprintf(fid, "\\end{tikzpicture}\n");

fclose (fid);
end