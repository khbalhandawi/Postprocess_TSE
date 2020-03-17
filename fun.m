function [ out ] = fun( X, extra_param )

if (nargin==2)
    param=extra_param;
end

P_analysis = param{1};
weight = param{2};
resiliance = param{3};

[tf, index]=ismember(P_analysis,X','rows');

if any(tf) % if branch is found
    W = weight(tf,:);
    R = resiliance(tf,:);
    f = -R;
else
    out = Inf;
end

out(1)=f;
  
end

