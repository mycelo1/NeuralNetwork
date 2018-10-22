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

        private bool initialized;

        public Network()
        {
            initialized = false;
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

            if (!(layers[0] is InputLayer))
            {
                throw new InvalidOperationException($"first layer must be {nameof(InputLayer)}");
            }

            for (int x = 1; x < layers.Count; x++)
            {
                layers[x].Initialize(RandomWeightRange);
            }

            initialized = true;
        }

        public double[] NetworkResults(double[] inputs)
        {
            if (!initialized)
            {
                throw new InvalidOperationException("network not initialized");
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
        public (int, double) Train(double[][] inputs, double[][] target_outputs, int epochs, double target_mean_sqr_error = 0)
        {
            double min_mean_sqr_error = Double.MaxValue;
            (int, double) result = (0, 0d);

            if (!initialized)
            {
                throw new InvalidOperationException("network not initialized");
            }

            if (inputs[0].Length != layers[0].Neurons)
            {
                throw new ArgumentOutOfRangeException($"{nameof(inputs)}");
            }

            if (target_outputs[0].Length != layers.Last().Neurons)
            {
                throw new ArgumentOutOfRangeException($"{nameof(target_outputs)}");
            }

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
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

                // calc mean squared error
                double sum_sqr_error = 0.0d;
                for (int row = 0; row < inputs.Length; row++)
                {
                    // send inputs throughout the network
                    double[] outputs = this.NetworkResults(inputs[row]);

                    for (int col = 0; col < outputs.Length; col++)
                    {
                        double error = target_outputs[row][col] - outputs[col];
                        sum_sqr_error += Math.Pow(error, 2);
                    }
                }

                double mean_sqr_error = sum_sqr_error / inputs.Length;

                // exit if mse < minimun
                if (mean_sqr_error < target_mean_sqr_error)
                {
                    result = (epoch, mean_sqr_error);
                    break;
                }

                // store optimal # of epochs
                if (mean_sqr_error < min_mean_sqr_error)
                {
                    min_mean_sqr_error = mean_sqr_error;
                    result = (epoch, mean_sqr_error);
                }
            }

            return result;
        }
    }
}