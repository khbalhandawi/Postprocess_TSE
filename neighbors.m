function [ Xout ] = neighbors ( X )  

c=X(1);
i0=X(2);
i1=X(3);
i2=X(4);
i3=X(5);

deposits = [0, 1 ,2 ,3];
% neighbor #1:
Xout(1,:) = [ c; i0 ; i1 ; i2 ; i3 ];

% neighbor #2:
Xout(2,:) = [ c; i0 ; i1 ; i2 ; i3 ];


if i1 == -1
    remove = [i0];
    for i = 1:1:length(remove)
        deposits = deposits(deposits~=remove(i));
    end
    population = deposits;
    i1_2 = randsample(population,1);
    
    % neighbor #1:
    Xout(1,:) = [ c; i0 ; i1_2 ; i2 ; i3 ];
    
elseif i1 ~= -1 && i2 == -1 && i3 == -1  
    i1_2 = -1;
    
    % neighbor #2:
    Xout(2,:) = [ c; i0 ; i1_2 ; i2 ; i3 ];
    
elseif i2 == -1
    remove = [i0, i1];
    for i = 1:1:length(remove)
        deposits = deposits(deposits~=remove(i));
    end
    population = deposits;
    i2_2 = randsample(population,1);
        
    % neighbor #1:
    Xout(1,:) = [ c; i0 ; i1 ; i2_2 ; i3 ];
    
elseif i2 ~= -1 && i3 == -1  
    i2_2 = -1;
        
    % neighbor #2:
    Xout(2,:) = [ c; i0 ; i1 ; i2_2 ; i3 ];
    
elseif i3 == -1
    remove = [i0, i1, i2];
    for i = 1:1:length(remove)
        deposits = deposits(deposits~=remove(i));
    end
    population = deposits;
    i3_2 = randsample(population,1);
        
    % neighbor #1:
    Xout(1,:) = [ c; i0 ; i1 ; i2 ; i3_2 ];
    
elseif i3 ~= -1
    i3_2 = -1;
        
    % neighbor #2:
    Xout(2,:) = [ c; i0 ; i1 ; i2 ; i3_2 ];
    
end
Xout
end
