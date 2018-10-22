using System;

using NeuralNetwork.Helpers;

namespace NeuralNetwork.Layers
{
    public abstract class Layer : ILayer
    {
        public ILayer PreviousLayer { get; set; }
        public ILayer NextLayer { get; set; }
        public int Neurons { get { return neurons; } }
        public double[] Inputs { get { return Inputs; } }
        public double[] Outputs { get { return outputs; } }

        public double[][] Weights;
        public double[] Biases;
        public double[] Gradients;

        protected double[][] PrevWeightsDelta;
        protected double[] PrevBiasesDelta;

        protected int neurons;
        protected double[] inputs;
        protected double[] outputs;

        public Layer(int num_neurons)
        {
            neurons = num_neurons;
        }

        public virtual void Initialize(double random_range)
        {
            inputs = new double[PreviousLayer.Neurons];
            outputs = new double[neurons];

            Weights = Helper.New2DArray<double>(PreviousLayer.Neurons, neurons);
            Biases = new double[neurons];
            Gradients = new double[neurons];

            PrevWeightsDelta = Helper.New2DArray<double>(PreviousLayer.Neurons, neurons);
            PrevBiasesDelta = new double[neurons];

            for (int x = 0; x < Weights.Length; x++)
            {
                for (int y = 0; y < Weights[0].Length; y++)
                {
                    Weights[x][y] = Helper.SmallRandom(random_range);
                }
            }

            for (int x = 0; x < Biases.Length; x++)
            {
                Biases[x] = Helper.SmallRandom(random_range);
            }
        }

        // compute layer input -> output
        public double[] LayerResults(double[] inputs)
        {
            Array.Copy(inputs, this.inputs, this.inputs.Length);

            if (inputs.Length != PreviousLayer.Neurons)
            {
                throw new ArgumentOutOfRangeException();
            }

            double[] sums = new double[this.Neurons];

            for (int this_neuron = 0; this_neuron < this.Neurons; this_neuron++)
            {
                sums[this_neuron] = Biases[this_neuron];
                for (int prev_neuron = 0; prev_neuron < PreviousLayer.Neurons; prev_neuron++)
                {
                    sums[this_neuron] += inputs[prev_neuron] * Weights[prev_neuron][this_neuron];
                }
            }

            double[] result = ActivactionFunction(sums);
            Array.Copy(result, outputs, outputs.Length);

            return result;
        }

        // find minimum errors to compute optimal weights
        public void UpdateWeights(double[] previous_outputs, double learn_rate, double momentum, double weight_decay, double bias)
        {
            for (int x = 0; x < Weights.Length; x++)
            {
                for (int y = 0; y < Weights[0].Length; y++)
                {
                    double delta = learn_rate * Gradients[y] * previous_outputs[x];
                    Weights[x][y] += delta;
                    Weights[x][y] += momentum * PrevWeightsDelta[x][y];
                    Weights[x][y] -= (weight_decay * Weights[x][y]);
                    PrevWeightsDelta[x][y] = delta;
                }
            }

            for (int x = 0; x < Biases.Length; x++)
            {
                double delta = learn_rate * Gradients[x] * bias;
                Biases[x] += delta;
                Biases[x] += momentum * PrevBiasesDelta[x];
                Biases[x] -= (weight_decay * Biases[x]);
                PrevBiasesDelta[x] = delta;
            }

            if (NextLayer != null)
            {
                ((Layer)NextLayer).UpdateWeights(this.outputs, learn_rate, momentum, weight_decay, bias);
            }
        }

        protected abstract double[] ActivactionFunction(double[] inputs);
        public abstract void ComputeGradients(double[] target_values, double[][] weights = null, double[] gradients = null);
    }
}