﻿public class Utility
{
    public static int Argmax(double[] values)
    {
        int argmax = 0;
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                argmax = i;
                max = values[i];
            }
        }
        return argmax;
    }

    public static double ArgmaxError(double[] prediction, List<Sample> samples)
    {
        int predictionArgmax = Argmax(prediction);
        int errorCount = 0;
        foreach (Sample sample in samples)
        {
            int sampleArgmax = Argmax(sample.output);
            if (sampleArgmax != predictionArgmax)
            {
                errorCount++;
            }
        }
        return (double)errorCount / (double)samples.Count;
    }

    public static double AbsoluteError(double[] a, double[] b)
    {
        double error = 0;
        for (int i = 0; i < a.Length; i++)
        {
            error += Math.Abs(a[i] - b[i]);
        }
        return error;
    }

    public static double EuclideanDistance(double[] a, double[] b)
    {
        double distance = 0;
        for (int i = 0; i < a.Length; i++)
        {
            distance += Math.Pow(a[i] - b[i], 2);
        }
        return Math.Sqrt(distance);
    }
}