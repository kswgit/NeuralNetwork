mutable struct NeuralNetwork
    lr::Float64  # learningrate
    wih::Array
    who::Array
end

function sigmoid(z)
    1.0 ./ (1.0 .+ exp.(-z))
end

function NewNeuralNetwork(inputnodes::Int64, 
                          hiddennodes::Int64, 
                          outputnodes::Int64, 
                          learningrate::Float64)
    wih = 2 * rand(hiddennodes, inputnodes) - 1
    who = 2 * rand(outputnodes, hiddennodes) - 1
    NeuralNetwork(learningrate, wih, who)
end

function train(ann::NeuralNetwork, input::Array, target::Array)
    hidden_output = sigmoid(ann.wih * input)
    output = sigmoid(ann.who * hidden_output)
    output_errors = target - output
    hidden_errors = transpose(ann.who) * output_errors
    ann.who += ann.lr .* (output_errors .* output .* (1.0 - output)) * transpose(hidden_output)
    ann.wih += ann.lr .* (hidden_errors .* hidden_output .* (1.0 - hidden_output)) * transpose(input)
    ann
end

function query(ann::NeuralNetwork, input)
    hidden_output = sigmoid(ann.wih * input)
    final_output = sigmoid(ann.who * hidden_output)
end

ann = NewNeuralNetwork(784, 100, 10, 0.3)
onodes = 10

data_file = open("../mnist_data/mnist_train_100.csv", "r")
data_list = readlines(data_file)
close(data_file)
test_data_file = open("../mnist_data/mnist_test_10.csv", "r")
test_data_list = readlines(test_data_file)
close(test_data_file)

for i=2:100
    all_values = split(data_list[i], ',')
    v = map(x -> parse(Int, x), all_values[2:785])
    inputs = v / 255.0 * 0.99 + 0.01
    targets = zeros(onodes) + 0.01
    targets[parse(Int, all_values[1]) + 1] = 0.99
    ann = train(ann, inputs, targets)
end
all_values = split(test_data_list[4], ',')
v = map(x -> parse(Int, x), all_values[2:785])
inputs = v / 255.0 * 0.99 + 0.01
guess = query(ann, inputs)
println("input: ", parse(Int, all_values[1]))
println("output ", indmax(guess) - 1)
println("detail")
for i in eachindex(guess)
    println(i - 1, ": ", round(guess[i] * 100), "%")
end
