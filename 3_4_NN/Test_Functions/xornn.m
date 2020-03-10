function layer2 = xornn(iters)
    if nargin < 1
        iters = 50
    end
    function s = sigmoid(X)
        s = 1.0 ./ (1.0 + exp(-X));
    end
    T = [0 1 1 0];
    X = [0 0 1 1; 0 1 0 1; 1 1 1 1];
    theta1 = [11 0 -5; 0 12 -7;18 17 -20];
    theta2 = [14 13 -28 -6];
    for i = [1:iters]
        layer1 = [sigmoid(theta1 * X); 1 1 1 1];
        layer2 = sigmoid(theta2 * layer1)
        delta2 = T - layer2;
        delta1 = layer1 .* (1-layer1) .* (theta2' * delta2);
        % remove the bias from delta 1. There's no real point in a delta on the bias.
        delta1 = delta1(1:3,:);
        theta2d = delta2 * layer1';
        theta1d = delta1 * X';
        theta1 = theta1 + 0.1 * theta1d;
        theta2 = theta2 + 0.1 * theta2d;
    end
end