# Prac 4 - Gradient Descent

using LinearAlgebra

function gradient_descent(P, q, x₀; α = 0.1, maxitter = 1000, ϵ = 1e-5)
    x = copy(x₀)
    ▽f = x -> P * x + q
    △x = -▽f(x)
    iter = 0
    while norm(△x) > ϵ || iter <= maxitter
        iter += 1
        x .+= ϵ * △x
        △x .= -▽f(x)
    end
    return x
end

P = [10.0 -1.0;
    -1.0 1.0];

q = [0; -10.0];

x₀ - zeros(2);

result = gradient_descent(P,q,x₀)


