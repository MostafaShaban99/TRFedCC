clc;
clear all;
close all;
%% Inputs
n_bus=33;
n_para=2;
n_sample=2*8760; %Two years

%% Measures for ECs

vol_ang=[23,24,25; 
                 26 27 28;
                 30 31 32;
                 3 4 5;
                 6 7 8;
                 10 11 12;
                 13 14 15;
                 16 17 18;
                 20 21 22];
             
p_q=[22 23 24;
          25 26 27;
          29 30 31;
          2 3 4;
          5 6 7;
          9 10 11;
          12 13 14;
          15 16 17;
          19 20 21];

%% Load Case
 for i=1:n_sample
     
     mpc=loadcase('case33bw');
     data=mpc.bus(:,3:4);
     error = min(max(0.5 *randn(size(data)), -1), 1);
     data1=data.*(1+error);
     data1(1,:)=data(1,:);
     mpc.bus(:,3:4)=data1;
     results=runopf(mpc); 

      injet_P=results.branch(:,14);
      injet_Q=results.branch(:,15);
      volt=results.bus(:,8);
      ang=results.bus(:,9);
      Data1=[ injet_P(p_q(1,:)), injet_Q(p_q(1,:)), volt(vol_ang(1,:)), ang(vol_ang(1,:))];
      Data1= [Data1(:,1)', Data1(:,2)', Data1(:,3)', Data1(:,4)'];
      
      Data2=[ injet_P(p_q(2,:)), injet_Q(p_q(2,:)), volt(vol_ang(2,:)), ang(vol_ang(2,:))];
      Data2= [Data2(:,1)', Data2(:,2)', Data2(:,3)', Data2(:,4)'];      
      
      Data3=[ injet_P(p_q(3,:)), injet_Q(p_q(3,:)), volt(vol_ang(3,:)), ang(vol_ang(3,:))];
      Data3= [Data3(:,1)', Data3(:,2)', Data3(:,3)', Data3(:,4)']; 
      
      Data4=[ injet_P(p_q(4,:)), injet_Q(p_q(4,:)), volt(vol_ang(4,:)), ang(vol_ang(4,:))];
      Data4= [Data4(:,1)', Data4(:,2)', Data4(:,3)', Data4(:,4)'];       
      
      Data5=[ injet_P(p_q(5,:)), injet_Q(p_q(5,:)), volt(vol_ang(5,:)), ang(vol_ang(5,:))];
      Data5= [Data5(:,1)', Data5(:,2)', Data5(:,3)', Data5(:,4)']; 
      
       Data6=[ injet_P(p_q(6,:)), injet_Q(p_q(6,:)), volt(vol_ang(6,:)), ang(vol_ang(6,:))];
      Data6= [Data6(:,1)', Data6(:,2)', Data6(:,3)', Data6(:,4)'];      
      
      Data7=[ injet_P(p_q(7,:)), injet_Q(p_q(7,:)), volt(vol_ang(7,:)), ang(vol_ang(7,:))];
      Data7= [Data7(:,1)', Data7(:,2)', Data7(:,3)', Data7(:,4)'];  
      
      Data8=[ injet_P(p_q(8,:)), injet_Q(p_q(8,:)), volt(vol_ang(8,:)), ang(vol_ang(8,:))];
      Data8= [Data8(:,1)', Data8(:,2)', Data8(:,3)', Data8(:,4)'];       
      
      Data9=[ injet_P(p_q(9,:)), injet_Q(p_q(9,:)), volt(vol_ang(9,:)), ang(vol_ang(9,:))];
      Data9= [Data9(:,1)', Data9(:,2)', Data9(:,3)', Data9(:,4)'];            
      
%     % Get the output Data
    outpd(i,:)=[Data1 Data2  Data3 Data4 Data5 Data6  Data7 Data8 Data9];
 
 end
 
  save('raw_data.mat', 'outpd', '-v7.3')


