using Flux, Plots


function myFun(x1, x2)
    f = x1 * exp(-x1^2-x2^2)
    return f
end

# Generate training and test data sets
Ntrain = 500
X_train = 4.0 * rand(2, Ntrain) .- 2.0

Ntest = 100
X_test = 4.0 * rand(2, Ntest) .- 2.0

Y_train = zeros(Ntrain)
for i in range(1,Ntrain)
    Y_train[i] = myFun(X_train[1,i],X_train[2,i])
end

Y_test = zeros(Ntest)
for i in range(1,Ntest)
    Y_test[i] = myFun(X_test[1,i],X_test[2,i])
end

# NN architecture
model = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 1))

# Optimizer
opt_state = Flux.setup(Adam(1e-3), model)

train_loss_vec = []
test_loss_vec = []
max_epoch = 5000
for epoch in 1:max_epoch
    # Compute the loss and the gradients:
    loss, gs = Flux.withgradient(m -> Flux.mae(m(X_train)[:], Y_train), model)
    # Update the model parameters (and the Adam momenta):
    Flux.update!(opt_state, model, gs[1])
    if mod(epoch, 5) == 0
        # Report on train and test, only every 2nd epoch:
        train_loss = Flux.mae(model(X_train)[:], Y_train)
        push!(train_loss_vec, train_loss)
        test_loss = Flux.mae(model(X_test)[:], Y_test)
        push!(test_loss_vec, test_loss)
        @info "After epoch = $epoch" train_loss test_loss
    end

end

plot(1:length(train_loss_vec), train_loss_vec, label="train", linewidth=2)
plot!(1:length(test_loss_vec), test_loss_vec, label="test", linewidth=2)
savefig("loss.png")

npoints = 50
x1 = range(-2, stop = 2, length = npoints)
x2 = range(-2, stop = 2, length = npoints)
x1 = x1' .* ones(npoints)
x2 = ones(npoints)' .* x2
surface(x1[:], x2[:], myFun.(x1[:], x2[:]))
scatter!(x1[:][1:2:end], x2[:][1:2:end], model([x1[:] x2[:]]')[:][1:2:end], label="NN", markersize=1)
savefig("function_approximation.png")