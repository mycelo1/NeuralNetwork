using System;
using System.Collections.Generic;

namespace NeuralNetwork.Helpers
{
    internal static class Helper
    {
        private static Random random = new Random();

        public static double SmallRandom(double resolution)
        {
            double min = -Math.Abs(resolution);
            double max = +Math.Abs(resolution);
            return (max - min) * random.NextDouble() + min;
        }

        public static T[][] New2DArray<T>(int num_x, int num_y)
        {
            T[][] result = new T[num_x][];

            for (int x = 0; x < result.Length; x++)
            {
                result[x] = new T[num_y];
            }

            return result;
        }

        public static void ZeroDoubleArray(double[] input)
        {
            for (int x = 0; x < input.Length; x++)
            {
                input[x] = 0;
            }
        }

        public static int[] ShuffledSequence(int length)
        {
            int[] result = new int[length];

            for (int x = 0; x < result.Length; x++)
            {
                result[x] = x;
            }

            for (int x = 0; x < result.Length; x++)
            {
                int random_pos = random.Next(x, result.Length);
                int current_value = result[random_pos];
                result[random_pos] = result[x];
                result[x] = current_value;
            }

            return result;
        }

        public static void Normalize(ref double[][] data)
        {
            for (int y = 0; y < data[0].Length; y++)
            {
                // calc mean
                double mean_sum = 0.0d;
                for (int x = 0; x < data.Length; ++x)
                {
                    mean_sum += data[x][y];
                }
                double mean = mean_sum / data.Length;

                // calc standard deviation
                double sd_sum = 0.0d;
                for (int x = 0; x < data.Length; ++x)
                {
                    sd_sum += (data[x][y] - mean) * (data[x][y] - mean);
                }
                double sd = Math.Sqrt(sd_sum / (data.Length - 1));

                // apply (x - mean) / sd
                for (int x = 0; x < data.Length; ++x)
                {
                    data[x][y] = (data[x][y] - mean) / sd;
                }
            }
        }

        public static T Last<T>(this List<T> list)
        {
            if (list.Count > 0)
            {
                return list[list.Count - 1];
            }
            else
            {
                return default(T);
            }
        }
    }
}
