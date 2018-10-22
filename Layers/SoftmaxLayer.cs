using System;

namespace NeuralNetwork.Layers
{
    // Softmax output layer
    public class SoftmaxLayer : Layer
    {
        public SoftmaxLayer(int num_neurons) : base(num_neurons)
        {
            //
        }

        public override void Initialize(double random_range)
        {
            if ((PreviousLayer == null) || (NextLayer != null))
            {
                throw new InvalidOperationException($"{nameof(SoftmaxLayer)} should be the last layer");
            }

            base.Initialize(random_range);
        }

        // Softmax activaction function (all results sum 1)
        protected override double[] ActivactionFunction(double[] w_sums)
        {
            double max = double.MinValue;
            double scale = 0.0;
            double[] result = new double[w_sums.Length];

            for (int x = 0; x < w_sums.Length; ++x)
            {
                if (w_sums[x] > max) max = w_sums[x];
            }

            for (int x = 0; x < w_sums.Length; ++x)
            {
                scale += Math.Exp(w_sums[x] - max);
            }

            for (int x = 0; x < w_sums.Length; ++x)
            {
                result[x] = Math.Exp(w_sums[x] - max) / scale;
            }

            return result;
        }

        // Softmax derivative = (1 - y) * y
        public override void ComputeGradients(double[] target_values, double[][] weights, double[] gradients)
        {
            double[] result = new double[this.Neurons];

            for (int x = 0; x < this.Neurons; x++)
            {
                result[x] = ((1 - outputs[x]) * outputs[x]) * (target_values[x] - outputs[x]);
            }

            Array.Copy(result, Gradients, Gradients.Length);
            ((Layer)PreviousLayer).ComputeGradients(null, Weights, Gradients);
        }
    }
}