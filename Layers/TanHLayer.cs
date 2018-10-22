using System;

namespace NeuralNetwork.Layers
{
    // TanH hidden layer
    public class TanHLayer : Layer
    {
        public TanHLayer(int num_neurons) : base(num_neurons)
        {
            //
        }

        public override void Initialize(double random_range)
        {
            if ((PreviousLayer == null) || (NextLayer == null))
            {
                throw new InvalidOperationException($"{nameof(TanHLayer)} should be a hidden layer");
            }

            base.Initialize(random_range);
        }

        // TanH activation function
        protected override double[] ActivactionFunction(double[] w_sums)
        {
            double[] result = new double[w_sums.Length];

            for (int x = 0; x < w_sums.Length; x++)
            {
                result[x] = w_sums[x] < -20d ? -1 : (w_sums[x] > +20d ? +1 : Math.Tanh(w_sums[x]));
            }

            return result;
        }

        // TanH derivative = (1 - y) * (1 + y)
        public override void ComputeGradients(double[] target_values, double[][] weights, double[] gradients)
        {
            double[] result = new double[this.Neurons];

            for (int x = 0; x < this.Neurons; x++)
            {
                double sum = 0.0d;
                for (int y = 0; y < gradients.Length; y++)
                {
                    sum += gradients[y] * weights[x][y];
                }

                result[x] = ((1 - outputs[x]) * (1 + outputs[x])) * sum;
            }

            Array.Copy(result, Gradients, Gradients.Length);

            if (!(PreviousLayer is InputLayer))
            {
                ((Layer)PreviousLayer).ComputeGradients(null, Weights, Gradients);
            }
        }
    }
}