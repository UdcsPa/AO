# use this---
            
using Flux

f(x) = 4x^2 + 3x + 2;

df(x) = gradient(f,x)[1];

abstract type DescentMEthod end

abstract type DescentMethod end

mutable struct
    Adagrad <: DescentMethod
       α # learning rate
       ϵ # small value
       s # sum of squared gradient
    end

function init!(M::Adagrad, f, df, x)
    M.s = zeros(length(x))
    return M
end

function step!(M::Adagrad, f, df, x)
    α, ϵ, s, g = M.α, M.ϵ, M.s, df(x)
    s = g*g
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end

M = Adagrad(0.01, 0.0001, 0.01)
Adagrad(0.01, 0.0001, 0.01)

result = step!(M, f, df, 2)

# 1.990000052631302

#----------------------------------------------

# RMSProp
mutable struct RMSProp <: DescentMethod
    α # learning rate
    γ # decay
    ϵ # small value
    s # sum of squared gradient
end
    
function init!(M::RMSProp, f, df, x)
    M.s = zeros(length(x))
    return M
end

function step!(M::RMSProp, f, df, x)
    α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, df(x)
    #s[:] = γ*s + (1-γ)*(g.*g)
    s = γ*s + (1-γ)*(g*g)
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end
 
using Flux

f(x) = 4x^2 + 3x + 2;
    
df(x) = gradient(f, x)[1];

#----------------------------------------------

# Adadelta
mutable struct Adadelta <: DescentMethod
    γs # gradient decay
    γx # update decay
    ϵ # small value
    s # sum of squared gradients
    u # sum of squared updates
end
    
function init!(M::Adadelta, f, df, x)
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M
end

function step!(M::Adadelta, f, df, x)
    γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, df(x)
    #s[:] = γs*s + (1-γs)*g.*g
    s = γs*s + (1-γs)*g.*g
    Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
    #u[:] = γx*u + (1-γx)*Δx.*Δx
    u = γx*u + (1-γx)*Δx.*Δx
    return x + Δx
end

#-----------------------------------
#=

# Quasi_Newton
f(x)=(x-3.14)^4
df(x)=4*(x-3.14)^3
fdf(x)=f(x),df(x)

abstract type DescentMethod end

mutable struct DFP <: DescentMethod
    Q
end

function init!(M::DFP, f, df, x)
    m = length(x)
    M.Q = Array{Float64,2}(undef,m,m)
    return M
end

a=f(0)
b=df(0)

using LineSearches

function step!(M::DFP, f, df, x)
    Q, g = M.Q, df(x)
    x′ =(Static())(f,df,fdf,9.0,a,b)
    g′ = df(x′[2])
    δ = x′[2] - x
    γ = g′ - g
    Q = Q[1] - Q[1]*γ*γ'*Q[1]/(γ*Q[1]*γ') + δ*δ'/(δ'*γ)
    return x′
end

=#



