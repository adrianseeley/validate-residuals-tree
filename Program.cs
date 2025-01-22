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

    public static List<double[]> KNN(List<Sample> train, double[] testInput, int maxK, double powerExponent, double rootExponent)
    {
        // find neighbours
        (Sample trainSample, double distance)[] neighboursArray = new (Sample trainSample, double distance)[train.Count];
        Parallel.For(0, train.Count, trainIndex =>
        {
            double[] trainInput = train[trainIndex].input;
            double distance = 0f;
            for (int i = 0; i < trainInput.Length; i++)
            {
                distance += Math.Pow(Math.Abs(trainInput[i] - testInput[i]), powerExponent);
            }
            distance = Math.Pow(distance, 1f / rootExponent);
            neighboursArray[trainIndex] = (train[trainIndex], distance);
        });

        // sort near to far
        List<(Sample trainSample, double distance)> neighbours = neighboursArray.ToList();
        neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));

        // create results
        List<double[]> results = new List<double[]>();

        // iterate through k values
        for (int k = 1; k <= maxK; k++)
        {
            // get the k neighbours
            List<(Sample trainSample, double distance)> kNeighbours = neighbours.Take(k).ToList();

            // find the max distance of the neighbours
            double maxDistance = kNeighbours.Last().distance;

            // create the result and weight sum
            double[] result = new double[train[0].output.Length];
            double weightSum = 0f;

            // iterate through neighbours
            foreach ((Sample trainSample, double distance) kNeighbour in kNeighbours)
            {
                // calculate the weight of this neighbour
                double weight = maxDistance / kNeighbour.distance;

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
            }

            // add to results
            results.Add(result);
        }

        // return results
        return results;
    }

    public static void Main(string[] args)
    {
        List<Sample> train = Data.MNIST("d:/data/mnist_train.csv");
        List<Sample> test = Data.MNIST("d:/data/mnist_test.csv");

        TextWriter log = new StreamWriter("log.csv");
        log.WriteLine("k,exponent,argmax,mae");
        log.Flush();

        int maxK = 100;
        for (double exponent = 1f; exponent < 30f; exponent += 0.1f)
        {
            Console.WriteLine("Exponent: " + exponent);
            int[] kArgmax = new int[maxK];
            double[] kMAE = new double[maxK];
            for (int sampleIndex = 0; sampleIndex < test.Count; sampleIndex++)
            {
                Console.Write($"\rTest: {sampleIndex + 1}/{test.Count}");
                Sample testSample = test[sampleIndex];
                List<double[]> predictions = KNN(train, testSample.input, maxK, exponent, exponent);
                double[] actual = testSample.output;
                int actualArgmax = Argmax(actual);
                for (int k = 1; k <= maxK; k++)
                {
                    double[] kPrediction = predictions[k - 1];
                    int kArgmaxPrediction = Argmax(kPrediction);
                    if (kArgmaxPrediction == actualArgmax)
                    {
                        kArgmax[k - 1]++;
                    }
                    for (int i = 0; i < kPrediction.Length; i++)
                    {
                        kMAE[k - 1] += Math.Abs(kPrediction[i] - actual[i]);
                    }
                }
            }
            Console.WriteLine();
            for (int k = 1; (k <= maxK); k++)
            {
                kMAE[k - 1] /= ((double)test.Count * (double)test[0].output.Length);
            }
            for (int k = 1; k <= maxK; k++)
            {
                Console.WriteLine($"K: {k}, Exponent: {exponent}, Argmax: {kArgmax[k - 1]}, MAE: {kMAE[k - 1]}");
                log.WriteLine($"{k},{exponent},{kArgmax[k - 1]},{kMAE[k - 1]}");
                log.Flush();
            }
        }
    }
}