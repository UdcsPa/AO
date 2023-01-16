using GaussianProcesses
using Random

Random.seed!(20140430)
# Training data
n = 10;                         # number of training points
x = 2π* rand(n);                # predictors
y = sin.(x) + 0.05*randn(n);    # regressors

# Selec mean and covariance function
mZero = MeanZero()              # Zero mean function
kern = SE(0.0,0.0)              # Squared exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0

# log standard deviation of observation noise (this is optimal)
gp = GP(x, y, mZero, kern, logObsNoise)

x = 0:0.1:2π
plot(gp; obsv = false)
optimize!(gp)
plot(gp; obsv = false, label = "GP posterior mean", fmt =:png)
samples = rand(gp, x, 5)
plot!(x, samples)



