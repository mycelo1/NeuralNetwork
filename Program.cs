using System;

using NeuralNetwork;
using NeuralNetwork.Helpers;
using NeuralNetwork.Layers;

public class Program
{
    public static void Main(string[] args)
    {
        double[][] train_data;
        double[][] train_target;
        double[][] test_data;
        double[][] test_target;

        // get 80% for train data and 20% for testing data
        TrainData(out train_data, out train_target, out test_data, out test_target);

        // normalize for faster results???
        //Helper.Normalize(ref train_data);
        //Helper.Normalize(ref test_data);

        // create network
        Network ann = new Network();
        ann
            .AddLayer(new InputLayer(4))        // 4-neurons input layer
            .AddLayer(new TanHLayer(7))         // 7-neurons hidden layer
            .AddLayer(new SoftmaxLayer(3));     // 3-neurons output layer

        // initialize random weights and biases
        ann.Initialize();

        // training
        ann.Train(train_data, train_target, 10000, 0.02d);

        // test accuracy on training data and new data
        Console.WriteLine(ann.TestAccuracy(train_data, train_target).ToString("F6"));
        Console.WriteLine(ann.TestAccuracy(test_data, test_target).ToString("F6"));
    }

    // slice dataset vertically and horizontally
    private static void TrainData(out double[][] train_data, out double[][] train_target, out double[][] test_data, out double[][] test_target)
    {
        // Iris Flower Classification data set
        double[][] all_data = new double[150][];
        all_data[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
        all_data[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 };
        all_data[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
        all_data[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
        all_data[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
        all_data[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
        all_data[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
        all_data[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
        all_data[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
        all_data[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

        all_data[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
        all_data[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
        all_data[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
        all_data[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
        all_data[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
        all_data[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
        all_data[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
        all_data[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
        all_data[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
        all_data[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };

        all_data[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
        all_data[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
        all_data[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
        all_data[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
        all_data[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
        all_data[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
        all_data[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
        all_data[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
        all_data[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
        all_data[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };

        all_data[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
        all_data[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
        all_data[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
        all_data[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
        all_data[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
        all_data[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
        all_data[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
        all_data[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
        all_data[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
        all_data[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };

        all_data[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
        all_data[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
        all_data[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
        all_data[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
        all_data[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
        all_data[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
        all_data[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
        all_data[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
        all_data[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
        all_data[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };

        all_data[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
        all_data[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
        all_data[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
        all_data[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
        all_data[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
        all_data[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
        all_data[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
        all_data[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
        all_data[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
        all_data[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

        all_data[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
        all_data[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
        all_data[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
        all_data[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
        all_data[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
        all_data[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
        all_data[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
        all_data[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
        all_data[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
        all_data[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };

        all_data[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
        all_data[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
        all_data[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
        all_data[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
        all_data[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
        all_data[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
        all_data[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
        all_data[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
        all_data[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
        all_data[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };

        all_data[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
        all_data[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
        all_data[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
        all_data[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
        all_data[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
        all_data[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
        all_data[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
        all_data[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
        all_data[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
        all_data[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };

        all_data[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
        all_data[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
        all_data[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
        all_data[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
        all_data[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
        all_data[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
        all_data[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
        all_data[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
        all_data[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
        all_data[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };

        all_data[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
        all_data[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
        all_data[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
        all_data[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
        all_data[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
        all_data[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
        all_data[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
        all_data[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
        all_data[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
        all_data[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

        all_data[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
        all_data[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
        all_data[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
        all_data[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
        all_data[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
        all_data[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
        all_data[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
        all_data[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
        all_data[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
        all_data[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };

        all_data[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
        all_data[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
        all_data[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
        all_data[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
        all_data[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
        all_data[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
        all_data[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
        all_data[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
        all_data[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
        all_data[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };

        all_data[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
        all_data[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
        all_data[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
        all_data[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
        all_data[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
        all_data[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
        all_data[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
        all_data[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
        all_data[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
        all_data[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };

        all_data[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
        all_data[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
        all_data[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
        all_data[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
        all_data[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
        all_data[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
        all_data[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
        all_data[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
        all_data[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
        all_data[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

        int train_slice = (int)Math.Round(all_data.Length * 0.8);
        int test_slice = all_data.Length - train_slice;

        train_data = new double[train_slice][];
        train_target = new double[train_slice][];
        test_data = new double[test_slice][];
        test_target = new double[test_slice][];

        int[] sequence = Helper.ShuffledSequence(all_data.Length);
        for (int x = 0; x < sequence.Length; x++)
        {
            double[] row = all_data[sequence[x]];
            double[] data_row = new double[4];
            double[] result_row = new double[3];

            Array.Copy(row, 0, data_row, 0, 4);
            Array.Copy(row, 4, result_row, 0, 3);

            if (x < train_slice)
            {
                train_data[x] = data_row;
                train_target[x] = result_row;
            }
            else
            {
                test_data[x - train_slice] = data_row;
                test_target[x - train_slice] = result_row;
            }
        }
    }
}