using System;

namespace NeuralNetwork.Layers
{
    public class InputLayer : ILayer
    {
        public ILayer PreviousLayer { get; set; }
        public ILayer NextLayer { get; set; }
        public int Neurons { get { return neurons; } }
        public double[] Inputs { get { return outputs; } }
        public double[] Outputs { get { return outputs; } }

        protected int neurons;
        protected double[] outputs;

        public InputLayer(int num_neurons)
        {
            neurons = num_neurons;
            outputs = new double[num_neurons];
        }

        public void Initialize(double random_range)
        {
            if ((PreviousLayer != null) || (NextLayer == null))
            {
                throw new InvalidOperationException($"{nameof(InputLayer)} should be the first layer");
            }
        }

        public double[] LayerResults(double[] inputs)
        {
            double[] result = new double[outputs.Length];
            Array.Copy(inputs, outputs, outputs.Length);
            Array.Copy(inputs, result, result.Length);
            return result;
        }
    }
}