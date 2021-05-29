%{%
files = {'data/summit/weakscale_summitV100N1n1.out';
         'data/summit/weakscale_summitV100N1n2.out';
         'data/summit/weakscale_summitV100N1n4.out';
         'data/summit/weakscale_summitV100N1n6.out';
         'data/summit/weakscale_summitV100N2n12.out';
         'data/summit/weakscale_summitV100N3n18.out';
         'data/summit/weakscale_summitV100N4n24.out';
         'data/summit/weakscale_summitV100N6n36.out';
         'data/summit/weakscale_summitV100N8n48.out'};
%}
%{
files = {'data/redwood/weakscale_redwoodMI100N1n1.out';
         'data/redwood/weakscale_redwoodMI100N1n2.out';
         'data/redwood/weakscale_redwoodMI100N1n4.out';
         'data/redwood/weakscale_redwoodMI100N2n8.out';
         'data/redwood/weakscale_redwoodMI100N3n12.out';
         'data/redwood/weakscale_redwoodMI100N4n16.out';
         'data/redwood/weakscale_redwoodMI100N5n20.out';
         'data/redwood/weakscale_redwoodMI100N6n24.out';
         'data/redwood/weakscale_redwoodMI100N7n28.out'};
%}       
         
         
Ndata=0;
clear data;

for f=1:size(files,1)
  
  fid = fopen(files{f},'rt');

  line=0;

  while true
    thisline = fgetl(fid);
    line++;
    
    if ~ischar(thisline); break; end  %end of file
     
    splitline = strsplit(thisline, {",",":"});
    
    if (size(splitline,2)==9)
      offset=0;  
    elseif (size(splitline,2)==10)
      offset=1;
    elseif (size(splitline,2)==11)
      offset=2;
    else 
      splitline
    end
    
    if strncmp(splitline{1},"AR", 2) 
      exchange=1;
    elseif strncmp(splitline{1},"PW", 2) 
      exchange=2;
    elseif strncmp(splitline{1},"CR", 2) 
      exchange=3;
    else 
      splitline{1}
    end
    
    overlap=false;
    gpuaware=false;
    if strncmp(splitline{2}," Overlap", 8) 
      overlap=true;
    elseif strncmp(splitline{2}," GPU-aware", 10) 
      gpuaware=true;
      if strncmp(splitline{3}," Overlap", 8) 
        overlap=true;
      end
    end
    
    rankStr = strsplit(splitline{2+offset});
    Nranks = str2num(rankStr{size(rankStr,2)});
    
    %skip single device data
    if (Nranks==1); continue; end;
    
    gDofsStr = strsplit(splitline{3+offset});
    NGdofs = str2num(gDofsStr{size(gDofsStr,2)});
    
    lDofsStr = strsplit(splitline{4+offset});
    NLdofs = str2num(lDofsStr{size(lDofsStr,2)});
    
    pStr = strsplit(splitline{5+offset});
    p = str2num(pStr{size(pStr,2)});
    
    %skip non p=3 data
    if (p~=3); continue; end;
    
    NvecStr = strsplit(splitline{6+offset});
    Nvec = str2num(NvecStr{size(NvecStr,2)});
    
    %skip Nvector hiding tests
    %if (Nvec~=0); continue; end;
    
    timeStr = strsplit(splitline{7+offset});
    time = str2num(timeStr{size(timeStr,2)-1});
    
    dpsStr = strsplit(splitline{8+offset});
    dps = str2num(dpsStr{size(dpsStr,2)});
    
    dprsStr = strsplit(splitline{9+offset});
    dprs = str2num(dprsStr{size(dprsStr,2)});
    
    Ndata++;
    data(Ndata) = struct("Exchange", exchange,
                         "Overlap", overlap,
                         "GPU-aware", gpuaware,
                         "Nranks", Nranks,
                         "Global_Dofs", NGdofs,
                         "Local_Dofs", NLdofs,
                         "Degree", p,
                         "Nvectors", Nvec,
                         "Time", time,
                         "Dofs/s", dps,
                         "Dofs/rank*s", dprs);
  end

  fclose(fid);
end
