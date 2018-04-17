include("datasets.jl")


G_training_dataset = get_addition_dataset(5000)

G_nr_batches = 100
G_nr_runs = 50
G_nr_updates = round(Int64, length(G_training_dataset)/G_nr_batches) * G_nr_runs

function neural_net()
    #=
    # Setup network
    =#

    layers  = [2, 900, 900, 900, 6 * 3]
    biases  = map(x -> randn(x) , layers)
    weights = map(i -> randn(layers[i], layers[i + 1]) , keys(layers)[1:length(layers) - 1])

    # Initial learning rate
    α = 0.001

    # Probability of removing neuron
    dropout = 0.3

    # Momentum factor
    ρ = 0.7

    sigmoid(x)  = 1.0./(1.0+exp(-x))
    softplus(x) = log(1.0+exp(x))
    softmax(x)  = exp(x)./sum(exp(x))

    δsigmoid(y)     = y.*(1.0-y)
    δsoftplus(y)    = 1.0-exp(-y)
    δsoftmax(y)     = y.*(1.0-y)

    remove_neuron(n) = 1.0*(dropout .<= rand(n))

    #=
    # Train network
    =#
    δbiases  = map(x -> zeros(x) , layers)
    δweights = map(i -> zeros(layers[i], layers[i + 1]) , keys(layers)[1:length(layers) - 1])

    nr_correct  = Int64(0)
    nr_tries    = Int64(0)

    for iu in range(1, G_nr_updates)

        remove = remove_neuron(biases[2:length(biases) - 1]) *1.0

        for ib in range(1, G_nr_batches)

            # Load a random element from the dataset
            k = rand(1:length(G_training_dataset))

            x = G_training_dataset[k][1:2]

            z = zeros(layers[length(layers)])

            # Feedforward pass for computing the output
            y = Vector()
            push!(y, sigmoid(x + biases[1]))
            push!(y, softplus(weights[1] * y[1] + biases[2]) * remove[1])
            push!(y, softplus(weights[2] * y[2] + biases[3]) * remove[2])
            push!(y, softplus(weights[3] * y[3] + biases[4]) * remove[3])
            push!(y, softmax(weights[4] * y[4] + biases[5]))

            # Backprop
            errors = Vector()

            push!(errors, z - y[5])
            push!(errors, weights[4] * δsoftplus(y[4]) * remove[3] )
            push!(errors, weights[3] * δsoftplus(y[3]) * remove[2] )
            push!(errors, weights[2] * δsoftplus(y[2]) * remove[1] )
            push!(errors, weights[1] * δsigmoid(y[1]) * weights[1] * e[4] )
            errors = reverse(errors)

			δbiases[1] += errors[1]
			δweights[1] += y1*errors[2]
			δbiases[2] += errors[2]
			δweights[2] += y2*errors[3]
			δbiases[3] += errors[3]
			δweights[3] += y3*errors[4]
			δbiases[4] += errors[4]
			δweights[4] += y4*errors[5]
			δbiases[5] += errors[5]

        end
    end
end
