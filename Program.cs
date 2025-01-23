﻿using static System.Net.Mime.MediaTypeNames;
using System.Diagnostics;

public class Program
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

    public static List<double[]> KNN(int maxK, List<Sample> train, double[] inputWeights, double[] distanceWeights, double distanceExponent, double distanceRoot, double weightExponent, double[] testInput, int ignoreTrainIndex = -1)
    {
        // find neighbours
        (int trainIndex, Sample trainSample, double distance)[] neighboursArray = new (int trainIndex, Sample trainSample, double distance)[train.Count];
        Parallel.For(0, train.Count, trainIndex =>
        {
            Sample trainSample = train[trainIndex];
            if (trainIndex == ignoreTrainIndex)
            {
                neighboursArray[trainIndex] = (trainIndex, trainSample, double.MaxValue);
                return;
            }
            double[] trainInput = train[trainIndex].input;
            double distance = 0f;
            for (int inputIndex = 0; inputIndex < trainInput.Length; inputIndex++)
            {
                distance += Math.Pow(Math.Abs(trainInput[inputIndex] - testInput[inputIndex]), distanceExponent) * inputWeights[inputIndex];
            }
            distance = Math.Pow(distance, 1f / distanceRoot) * distanceWeights[trainIndex];
            neighboursArray[trainIndex] = (trainIndex, trainSample, distance);
        });

        // sort near to far
        List<(int trainIndex, Sample trainSample, double distance)> neighbours = neighboursArray.ToList();
        neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));

        // create results
        List<double[]> results = new List<double[]>();

        // iterate through k values
        for (int k = 1; k <= maxK; k++)
        {
            // get the k neighbours
            List<(int trainIndex, Sample trainSample, double distance)> kNeighbours = neighbours.Take(k).ToList();

            // find the max distance of the neighbours
            double maxDistance = kNeighbours.Last().distance;

            // create the result and weight sum
            double[] result = new double[train[0].output.Length];
            double weightSum = 0f;

            // if any neighbours are 0 its a special case
            if (kNeighbours[0].distance == 0.0)
            {
                // gather all 0 distance neighbours
                double zeroDistanceWeightSum = 0f;
                foreach ((int trainIndex, Sample trainSample, double distance) kNeighbour in kNeighbours)
                {
                    // break once we hit a non 0 distance
                    if (kNeighbour.distance != 0.0)
                    {
                        break;
                    }

                    // add to result
                    double weight = distanceWeights[kNeighbour.trainIndex];
                    zeroDistanceWeightSum += weight;
                    for (int i = 0; i < result.Length; i++)
                    {
                        result[i] += kNeighbour.trainSample.output[i] * weight;
                    }
                }

                // weighted average result
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] /= zeroDistanceWeightSum;
                }

                // add to results
                results.Add(result);

                // continue to next k
                continue;
            }

            // reaching here implies no 0 distance neighbours

            // iterate through neighbours
            foreach ((int trainIndex, Sample trainSample, double distance) kNeighbour in kNeighbours)
            {
                // calculate the weight of this neighbour
                double weight;
                if (k == 1)
                {
                    weight = 1.0;
                }
                else
                {
                    weight = Math.Pow(maxDistance / kNeighbour.distance, weightExponent);
                }

                // add to weight sum
                weightSum += weight;

                // add to result
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] += kNeighbour.trainSample.output[i] * weight;
                }
            }

            // weighted average result
            for (int i = 0; i < result.Length; i++)
            {
                result[i] /= weightSum;
                if (double.IsNaN(result[i]))
                {
                    result[i] = 0.0; // nan/infinity can occur from exponent exploding
                }
            }

            // add to results
            results.Add(result);
        }

        // return results
        return results;
    }

    public static (int[] kArgmaxs, double[] kAbsoluteErrors) Score(int maxK, List<Sample> samples, int[] samplesArgmax, double[] inputWeights, double[] distanceWeights, double distanceExponent, double distanceRoot, double weightExponent)
    {
        int[] kArgmaxs = new int[maxK];
        double[] kAbsoluteErrors = new double[maxK];
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample testSample = samples[sampleIndex];
            List<double[]> predictions = KNN(maxK, samples, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent, testSample.input, sampleIndex);
            double[] actual = testSample.output;
            int actualArgmax = samplesArgmax[sampleIndex];
            for (int k = 1; k <= maxK; k++)
            {
                double[] kPrediction = predictions[k - 1];
                int kArgmaxPrediction = Argmax(kPrediction);
                if (kArgmaxPrediction == actualArgmax)
                {
                    kArgmaxs[k - 1]++;
                }
                for (int i = 0; i < kPrediction.Length; i++)
                {
                    kAbsoluteErrors[k - 1] += Math.Abs(kPrediction[i] - actual[i]);
                }
            }
        }
        return (kArgmaxs, kAbsoluteErrors);
    }

    public static bool IsBetter(ref double bestAbsoluteError, ref int bestArgmaxCorrect, int[] kArgmaxs, double[] kAbsoluteErrors)
    {
        bool improved = false;
        for (int k = 0; k < kArgmaxs.Length; k++)
        {
            double absoluteError = kAbsoluteErrors[k];
            int argmaxCorrect = kArgmaxs[k];
            if (absoluteError < bestAbsoluteError)
            {
                bestAbsoluteError = absoluteError;
                bestArgmaxCorrect = argmaxCorrect;
                improved = true;
            }
        }
        return improved;
    }

    public static void Main(string[] args)
    {
        Random random = new Random();
        int maxK = 30;
        List<(string name, List<Sample> samples)> datasets = new List<(string name, List<Sample> samples)>();
        datasets.Add(("IRIS", Data.IRIS("./data/IRIS/iris.data")));

        foreach ((string name, List<Sample> samples) in datasets)
        {
            int[] samplesArgmax = samples.Select(samples => Argmax(samples.output)).ToArray();
            
            TextWriter log = new StreamWriter(name + ".csv");
            log.WriteLine("epoch,absoluteError,argmaxCorrect");
            log.Flush();

            double nudge = 0.1;
            double distanceExponent = 1.0;
            double distanceRoot = 1.0;
            double weightExponent = 1.0;
            double[] inputWeights = samples[0].input.Select(input => 1.0).ToArray();
            double[] distanceWeights = samples.Select(sample => 1.0).ToArray();

            long epoch = 0;
            bool improved = true;
            double bestAbsoluteError = double.MaxValue;
            int bestArgmaxCorrect = 0;
            int[] kArgmaxs;
            double[] kAbsoluteErrors;

            (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
            if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
            {
                log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                log.Flush();
                Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, start");
            }

            while (improved)
            {
                improved = false;

                // distance exponent down
                epoch++;
                distanceExponent -= nudge;
                (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);

                // if better, keep change
                if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                {
                    log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                    log.Flush();
                    Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance exponent down");
                    improved = true;
                }
                // down wasnt better
                else
                {
                    // undo change AND, distance exponent up
                    epoch++;
                    distanceExponent += 2 * nudge;
                    (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);

                    // if better, keep change
                    if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                    {
                        log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                        log.Flush();
                        Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance exponent up");
                        improved = true;
                    }
                    // up wasnt better either
                    else
                    {
                        // reset to start
                        distanceExponent -= nudge;
                    }
                }

                // distance root down
                epoch++;
                distanceRoot -= nudge;
                (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);

                // if better, keep change
                if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                {
                    log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                    log.Flush();
                    Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance root down");
                    improved = true;
                }
                // down wasnt better
                else
                {
                    // undo change AND, distance root up
                    epoch++;
                    distanceRoot += 2 * nudge;
                    (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                    // if better, keep change
                    if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                    {
                        log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                        log.Flush();
                        Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance root up");
                        improved = true;
                    }
                    // up wasnt better either
                    else
                    {
                        // reset to start
                        distanceRoot -= nudge;
                    }
                }

                // weight exponent down
                epoch++;
                weightExponent -= nudge;
                (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);

                // if better, keep change
                if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                {
                    log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                    log.Flush();
                    Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, weight exponent down");
                    improved = true;
                }
                // down wasnt better
                else
                {
                    // undo change AND, weight exponent up
                    epoch++;
                    weightExponent += 2 * nudge;
                    (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                    // if better, keep change
                    if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                    {
                        log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                        log.Flush();
                        Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, weight exponent up");
                        improved = true;
                    }
                    // up wasnt better either
                    else
                    {
                        // reset to start
                        weightExponent -= nudge;
                    }
                }

                // iterate input weights
                for (int inputIndex = 0; inputIndex < inputWeights.Length; inputIndex++)
                {
                    // input weight down
                    epoch++;
                    inputWeights[inputIndex] -= nudge;
                    (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                    // if better, keep change
                    if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                    {
                        log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                        log.Flush();
                        Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, input weight down");
                        improved = true;
                    }
                    // down wasnt better
                    else
                    {
                        // undo change AND, input weight up
                        epoch++;
                        inputWeights[inputIndex] += 2 * nudge;
                        (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                        // if better, keep change
                        if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                        {
                            log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                            log.Flush();
                            Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, input weight up");
                            improved = true;
                        }
                        // up wasnt better either
                        else
                        {
                            // reset to start
                            inputWeights[inputIndex] -= nudge;
                        }
                    }
                }

                // iterate distance weights
                for (int distanceIndex = 0; distanceIndex < distanceWeights.Length; distanceIndex++)
                {
                    // distance weight down
                    epoch++;
                    distanceWeights[distanceIndex] -= nudge;
                    (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                    // if better, keep change
                    if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                    {
                        log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                        log.Flush();
                        Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance weight down");
                        improved = true;
                    }
                    // down wasnt better
                    else
                    {
                        // undo change AND, distance weight up
                        epoch++;
                        distanceWeights[distanceIndex] += 2 * nudge;
                        (kArgmaxs, kAbsoluteErrors) = Score(maxK, samples, samplesArgmax, inputWeights, distanceWeights, distanceExponent, distanceRoot, weightExponent);
                        // if better, keep change
                        if (IsBetter(ref bestAbsoluteError, ref bestArgmaxCorrect, kArgmaxs, kAbsoluteErrors))
                        {
                            log.WriteLine($"{epoch},{bestAbsoluteError},{bestArgmaxCorrect}");
                            log.Flush();
                            Console.WriteLine($"epoch: {epoch}, absolute: {bestAbsoluteError}, argmax: {bestArgmaxCorrect}, distance weight up");
                            improved = true;
                        }
                        // up wasnt better either
                        else
                        {
                            // reset to start
                            distanceWeights[distanceIndex] -= nudge;
                        }
                    }
                }
            }
   
            log.Close();
        }
    }
}