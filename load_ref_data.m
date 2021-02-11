%{
files = {'data/summit/ref/weakscale_summitV100N1n2_ref.out';
         'data/summit/ref/weakscale_summitV100N1n4_ref.out';
         'data/summit/ref/weakscale_summitV100N1n6_ref.out';
         'data/summit/ref/weakscale_summitV100N2n12_ref.out';
         'data/summit/ref/weakscale_summitV100N3n18_ref.out';
         'data/summit/ref/weakscale_summitV100N4n24_ref.out';
         'data/summit/ref/weakscale_summitV100N6n36_ref.out';
         'data/summit/ref/weakscale_summitV100N8n48_ref.out'};
%}
files = {'data/redwood/ref/weakscale_redwoodMI100N1n2_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N1n4_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N2n8_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N3n12_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N4n16_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N5n20_ref.out';
         'data/redwood/ref/weakscale_redwoodMI100N6n24_ref.out';};
         
Ndata=0;
clear refdata;

for f=1:size(files,1)
  
  fid = fopen(files{f},'rt');

  line=0;

  while true
    thisline = fgetl(fid);
    line++;
    
    if ~ischar(thisline); break; end  %end of file
     
    splitline = strsplit(thisline, {",",":"});
    
    rankStr = strsplit(splitline{1});
    Nranks = str2num(rankStr{size(rankStr,2)});
    
    %skip single device data
    if (Nranks==1); continue; end;
    
    gDofsStr = strsplit(splitline{2});
    NGdofs = str2num(gDofsStr{size(gDofsStr,2)});
    
    lDofsStr = strsplit(splitline{3});
    NLdofs = str2num(lDofsStr{size(lDofsStr,2)});
    
    pStr = strsplit(splitline{4});
    p = str2num(pStr{size(pStr,2)});
    
    %skip non p=3 data
    if (p~=3); continue; end;
  
    timeStr = strsplit(splitline{5});
    time = str2num(timeStr{size(timeStr,2)-1});
    
    dpsStr = strsplit(splitline{6});
    dps = str2num(dpsStr{size(dpsStr,2)});
    
    dprsStr = strsplit(splitline{7});
    dprs = str2num(dprsStr{size(dprsStr,2)});
    
    Ndata++;
    refdata(Ndata) = struct("Nranks", Nranks,
                         "Global_Dofs", NGdofs,
                         "Local_Dofs", NLdofs,
                         "Degree", p,
                         "Time", time,
                         "Dofs/s", dps,
                         "Dofs/rank*s", dprs);
  end

  fclose(fid);
end
