using System;
using System.Collections.Generic;

using NeuralNetwork.Helpers;
using NeuralNetwork.Layers;

// based on the work of Quaetrix Analytics

namespace NeuralNetwork
{
    public class Network
    {
        protected List<ILayer> layers = new List<ILayer>();

        public double WeightBias { get; set; } = 1.0d;
        public double WeightDecay { get; set; } = 0.0001d;
        public double LearnRate { get; set; } = 0.05d;
        public double Momentum { get; set; } = 0.01d;
        public double RandomWeightRange { get; set; } = 0.01d;

        public Network()
        {
            //
        }

        public Network AddLayer(ILayer layer)
        {
            if (layers.Count > 0)
            {
                layers.Last().NextLayer = layer;
                layer.PreviousLayer = layers.Last();
            }

            layers.Add(layer);
            return this;
        }

        public void Initialize()
        {
            if (layers.Count < 3)
            {
                throw new InvalidOperationException("must have at least 3 layers");
            }

            for (int x = 1; x < layers.Count; x++)
            {
                layers[x].Initialize(RandomWeightRange);
            }
        }

        public double[] NetworkResults(double[] inputs)
        {
            if (layers.Count == 0)
            {
                throw new InvalidOperationException("no layers");
            }

            if (inputs.Length != layers[0].Neurons)
            {
                throw new ArgumentOutOfRangeException($"{nameof(inputs)}");
            }

            double[] outputs = inputs;

            foreach (ILayer layer in layers)
            {
                outputs = layer.LayerResults(outputs);
            }

            return outputs;
        }

        // back propagation
        public void Train(double[][] inputs, double[][] target_outputs, int epochs, double mean_sqr_error)
        {
            if (layers.Count == 0)
            {
                throw new InvalidOperationException("no layers");
            }

            if (inputs[0].Length != layers[0].Neurons)
            {
                throw new ArgumentOutOfRangeException($"{nameof(inputs)}");
            }

            if (target_outputs[0].Length != layers.Last().Neurons)
            {
                throw new ArgumentOutOfRangeException($"{nameof(target_outputs)}");
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // calc mean squared error
                double sum_squared_error = 0.0d;
                for (int row = 0; row < inputs.Length; row++)
                {
                    // send inputs throughout the network
                    double[] outputs = this.NetworkResults(inputs[row]);

                    for (int col = 0; col < outputs.Length; col++)
                    {
                        double error = target_outputs[row][col] - outputs[col];
                        sum_squared_error += Math.Pow(error, 2);
                    }
                }

                // exit if mse < minimun
                if ((sum_squared_error / inputs.Length) < mean_sqr_error)
                {
                    break;
                }

                // randomize training sets order
                int[] sequence = Helper.ShuffledSequence(inputs.Length);

                // for each training set (in random order)
                for (int x = 0; x < sequence.Length; x++)
                {
                    // send inputs throughout the network
                    this.NetworkResults(inputs[sequence[x]]);

                    // compute gradients (last to first layer - except input layer)
                    ((Layer)layers.Last()).ComputeGradients(target_outputs[sequence[x]]);

                    // compute weights (first to last layer - except input layer)
                    ((Layer)layers[1]).UpdateWeights(layers[0].Inputs, LearnRate, Momentum, WeightDecay, WeightBias);
                }
            }
        }

        public double TestAccuracy(double[][] test_data, double[][] target_values)
        {
            int hits = 0;

            for (int x = 0; x < test_data.Length; ++x)
            {
                double[] result = NetworkResults(test_data[x]);
                double[] target = target_values[x];

                double max_value = Double.MinValue;
                int max_index = 0;

                for (int y = 0; y < result.Length; y++)
                {
                    if (result[y] > max_value)
                    {
                        max_index = y;
                        max_value = result[y];
                    }
                }

                if (target[max_index] == 1.0d)
                {
                    hits++;
                }
            }

            return (double)hits / (double)(test_data.Length);
        }
    }
}