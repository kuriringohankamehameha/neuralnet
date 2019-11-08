#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

class TrainingData {
    public:
        TrainingData(string filename) {
            data_file.open(filename.c_str());
        }
        bool is_eof() {
            return data_file.eof();
        }
        void get_network_graph(vector<uint32_t> &network_graph);
        uint32_t get_next_input(vector<double> &input_vals);
        uint32_t get_target_outputs(vector<double> &target_outputs);
    private:
        ifstream data_file;
};

void TrainingData::get_network_graph(vector<uint32_t> &network_graph) {
    string line, label;
    getline(data_file, line);
    stringstream ss(line);
    ss >> label;
    if (this->is_eof() || label.compare("graph:") != 0) {
        abort();
    }
    
    while (!ss.eof()) {
        uint32_t n;
        ss >> n;
        network_graph.push_back(n);
    }
}

uint32_t TrainingData::get_next_input(vector<double> &input_vals)
{
    input_vals.clear();

    string line;
    getline(data_file, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            input_vals.push_back(oneValue);
        }
    }

    return input_vals.size();
}

uint32_t TrainingData::get_target_outputs(vector<double> &target_outputs)
{
    target_outputs.clear();

    string line;
    getline(data_file, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            target_outputs.push_back(oneValue);
        }
    }

    return target_outputs.size();
}

struct Connection {
    double weight;
    double delta_weight;
};

class ActivationFunction {
    public:
        ActivationFunction(string name) {
            if (name.compare("tanh") == 0) {
                type = 0; 
            }
            else if (name.compare("sigmoid") == 0) {
                type = 1;
            }
            else {
                type = 2;
            }
        }
        double eval_function(double input) {
            if (type == 0) {
                return tanh(input); 
            }
            else if (type == 1) {
                return 1.0 / (1.0 + exp(-input));
            }
            else {
                return -10000.0;
            }
        }
        double eval_grad(double input) {
            if (type == 0) {
                return 1.0 - (input * input);
            }
            else if (type == 1) {
                return input * (1 - input);
            }
            else {
                return -10000.0;
            }
        }
    private:
        uint32_t type;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
    public:
        Neuron(uint32_t num_outputs, uint32_t index, string name);
        void feed_forward(Layer& prev_layer);
        void set_output(double val) {
            output = val;
        }
        double get_output(void) {
            return output;
        }
        void eval_grad(double target_val) {
            double delta = target_val - output;
            grad = delta * fn.eval_grad(output);
        }
        void eval_hidden_grad(const Layer &next_layer) {
            double sum = 0.0;
            for (uint32_t i=0; i<next_layer.size()-1; i++) {
                sum += output_weights[i].weight * next_layer[i].grad;
            }
            grad = sum * fn.eval_grad(output);
        }
        void update_weights(Layer &prev_layer) {
            for (uint32_t i=0; i<prev_layer.size(); i++) {
                Neuron &neuron = prev_layer[i];
                double old_delta_weight = neuron.output_weights[n_index].delta_weight;
                double new_delta_weight = eta * neuron.get_output() * grad + alpha * old_delta_weight;
                neuron.output_weights[n_index].delta_weight = new_delta_weight;
                neuron.output_weights[n_index].weight += new_delta_weight;
            }
        }
    private:
        // Neuron has an output value
        // and weights connecting it to other neurons
        static double random_weight(void) {
            return rand() / double(RAND_MAX);
        }
        double output;
        vector<Connection> output_weights;
        uint32_t n_index;
        static double activation_function(double input);
        static double activation_function_delta(double input);
        double grad;
        static double eta;
        static double alpha;
        ActivationFunction fn;
};

double Neuron::eta = 0.1;
double Neuron::alpha = 0.5;

Neuron::Neuron(uint32_t num_outputs, uint32_t index, string name) : fn(name) {
    for (uint32_t i = 0; i < num_outputs; i++) {
        output_weights.push_back(Connection());
        output_weights.back().weight = random_weight();
    }
    n_index = index;
}

void Neuron::feed_forward(Layer &prev_layer) {
    double sum = 0.0;
    for (uint32_t i=0; i<prev_layer.size(); i++) {
        sum += (prev_layer[i].get_output()) * prev_layer[i].output_weights[n_index].weight;
    }
    output = fn.eval_function(sum);
}

double Neuron::activation_function(double input) {
    return tanh(input);
}

double Neuron::activation_function_delta(double input) {
    return 1.0 - input*input;
}

class NeuralNet {
    public:
        NeuralNet(const vector<uint32_t> &network_graph, string name);
        void feed_forward(const vector<double> &input_vals);
        void back_prop(const vector<double> &target_vals);
        void get_results(vector<double> &result_vals);
        double get_last_error(void) {
            return last_error;
        }
    private:
        vector<Layer> layers; // layers[layernum][neuronnum]
        double error;
        double last_error;
        double smoothening_factor;
};

NeuralNet::NeuralNet(const vector<uint32_t> &network_graph, string name) {
    uint32_t num_layers = network_graph.size();
    for (uint32_t layer_num = 0; layer_num < num_layers; layer_num ++) {
        // Create a new Layer per layer number
        layers.push_back(Layer());
        uint32_t num_outputs = (layer_num == network_graph.size() - 1) ? 0 : network_graph[layer_num + 1];

        //Fill it with values corresponding to those in neurons
        for (uint32_t neuron_num = 0; neuron_num <= network_graph[layer_num]; neuron_num++) {
            layers.back().push_back(Neuron(num_outputs, neuron_num, name));
            cout << "Made a neuron\n";
        }

        // Set the bias as 1.0
        layers.back().back().set_output(1.0);
    }
}

void NeuralNet::feed_forward(const vector<double> &input_vals) {
    assert(input_vals.size() == layers[0].size() - 1);

    for (uint32_t i=0; i<input_vals.size(); i++) {
        layers[0][i].set_output(input_vals[i]);
    }

    for (uint32_t i=1; i<layers.size(); i++) {
        Layer& prev_layer = layers[i-1];
        for (uint32_t j=0; j<layers[i].size()-1; j++) {
            layers[i][j].feed_forward(prev_layer);
        }
    }
}

void NeuralNet::back_prop(const vector<double> &target_vals) {
    Layer &output_layer = layers.back();
    error = 0.0;
    for (uint32_t i=0; i<output_layer.size() - 1; i++) {
        double delta = target_vals[i] - output_layer[i].get_output();
        error += delta * delta;
    }
    error /= output_layer.size() - 1;
    error = sqrt(error);

    last_error = (last_error * smoothening_factor + error) / (smoothening_factor + 1.0); 
    
    for (uint32_t i=0; i<output_layer.size()-1; i++) {
        output_layer[i].eval_grad(target_vals[i]);
    }

    for (uint32_t i=layers.size()-2; i>0; i--) {
        Layer& hidden_layer = layers[i];
        Layer& next_layer = layers[i+1];

        for (uint32_t j=0; j<hidden_layer.size(); j++) {
            hidden_layer[j].eval_hidden_grad(next_layer);
        }
    }

    for (uint32_t i=layers.size()-1; i>0; i--) {
        Layer &layer = layers[i];
        Layer &prev_layer = layers[i-1];

        for (uint32_t j=0; j<layers.size()-1; j++) {
            layer[j].update_weights(prev_layer);
        }
    }

}

void NeuralNet::get_results(vector<double> &result_vals) {
    result_vals.clear();
    for (uint32_t i=0; i<layers.back().size()-1; i++) {
        result_vals.push_back(layers.back()[i].get_output());
    }
}


void show_vectors(string label, vector<double> &v) {
	cout << label << " ";
	for(uint32_t i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}


int main() {
    TrainingData train_data("trainingData.txt");
    vector<uint32_t> network_graph;
    train_data.get_network_graph(network_graph);
    NeuralNet nn(network_graph, "tanh");

    vector<double> input_vals, target_vals, result_vals;
    	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while(!train_data.is_eof())
	{
		trainingPass++;
		cout << endl << "Pass " << trainingPass;

		if(train_data.get_next_input(inputVals) != network_graph[0])
			break;
		show_vectors(": Inputs :", inputVals);
		nn.feed_forward(inputVals);

		nn.get_results(resultVals);
		show_vectors("Outputs: ", resultVals);

		train_data.get_target_outputs(targetVals);
		show_vectors("Targets: ", targetVals);
		assert(targetVals.size() == network_graph.back());

		nn.back_prop(targetVals);

		cout << "Net recent average error: "
		     << nn.get_last_error() << endl;
	}

	cout << endl << "Done" << endl;
}
