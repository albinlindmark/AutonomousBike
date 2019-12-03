function X = interpol(A,B,x,relation)

% Switch-case for the interpolation
switch relation
    
    case 0
        X = A;
    case 1
        X = A + ((B-A)/-0.1)*x;
    case 2
        X = A + ((B-A)/0.1)*x;
end