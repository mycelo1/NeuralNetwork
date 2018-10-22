using System;

namespace NeuralNetwork.Layers
{
    public interface ILayer
    {
        ILayer PreviousLayer { get; set; }
        ILayer NextLayer { get; set; }
        int Neurons { get; }
        double[] Inputs { get; }
        double[] Outputs { get; }
        void Initialize(double random_range);
        double[] LayerResults(double[] inputs);
    }
}