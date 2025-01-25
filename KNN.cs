public static class KNN
{
    public static void Run(Dataset dataset, KNNConfiguration knnConfiguration)
    {
        Scores scores = new Scores();
        List<(Sample[] samples, int[] samplesArgmax, bool isTrain)> sets = new List<(Sample[] samples, int[] samplesArgmax, bool isTrain)> { (dataset.train, dataset.trainArgmax, true), (dataset.test, dataset.testArgmax, false) };

        for (double distanceBias = knnConfiguration.distanceBiasMin; distanceBias <= knnConfiguration.distanceBiasMax; distanceBias += knnConfiguration.distanceBiasStep)
        {
            for (double distanceExponent = knnConfiguration.distanceExponentMin; distanceExponent <= knnConfiguration.distanceExponentMax; distanceExponent += knnConfiguration.distanceExponentStep)
            {
                for (double distanceRoot = knnConfiguration.distanceRootMin; distanceRoot <= knnConfiguration.distanceRootMax; distanceRoot += knnConfiguration.distanceRootStep)
                {
                    Console.Write($"\rinputDistanceBias: {distanceBias}/{knnConfiguration.distanceBiasMax}, distanceExponent: {distanceExponent}/{knnConfiguration.distanceExponentMax}, distanceRoot: {distanceRoot}/{knnConfiguration.distanceRootMax}        ");
                    foreach ((Sample[] samples, int[] samplesArgmax, bool isTrain) set in sets)
                    {
                        for (int sampleIndex = 0; sampleIndex < set.samples.Length; sampleIndex++)
                        {
                            double[] testInput = set.samples[sampleIndex].input;
                            int correctArgmax = set.samplesArgmax[sampleIndex];

                            // parallel compute distances to each neighbour
                            Parallel.For(0, dataset.train.Length, trainIndex =>
                            {
                                // get the train sample at index
                                Sample trainSample = dataset.train[trainIndex];

                                // get the neighbour at index
                                Neighbour neighbour = knnConfiguration.neighbours[trainIndex];

                                // if this is the ignored sample (for leave one out self-optimization)
                                if (set.isTrain && trainIndex == sampleIndex)
                                {
                                    // set null
                                    neighbour.index = -1;
                                    neighbour.sample = null;
                                    neighbour.distance = double.PositiveInfinity;
                                    return;
                                }

                                // get the input for this train sample
                                double[] trainInput = dataset.train[trainIndex].input;

                                // compute the distance between the test and train inputs
                                double distance = distanceBias;
                                for (int inputIndex = 0; inputIndex < trainInput.Length; inputIndex++)
                                {
                                    // we start with a bias distance of 1 between all inputs (creates a leverage for exponents and weights, even for identical inputs)
                                    double inputDistance = 0;

                                    // next we add the absolute difference between the inputs (absolute so its always positive, and can be raised to any power)
                                    inputDistance += Math.Abs(trainInput[inputIndex] - testInput[inputIndex]);

                                    // then we raise the input distance to the distance exponent (allowing for an emphasis on closer or further distances)
                                    inputDistance = Math.Pow(inputDistance, distanceExponent);

                                    // this is added to the total distance
                                    distance += inputDistance;
                                }

                                // first we take the specified root of the distance (this is different than the distance exponent, to allow a different compression of the distance)
                                distance = Math.Pow(distance, 1f / distanceRoot);

                                // store the train index, sample, and distance
                                neighbour.index = trainIndex;
                                neighbour.sample = trainSample;
                                neighbour.distance = distance;
                            });

                            // sort neighbours near to far
                            knnConfiguration.neighbours = knnConfiguration.neighbours.OrderBy(neighbour => neighbour.distance).ToArray();

                            // create results
                            List<double[]> results = new List<double[]>();

                            // iterate through k values
                            for (int k = 1; k <= knnConfiguration.maxK; k++)
                            {
                                // get the k neighbours
                                List<Neighbour> kNeighbours = knnConfiguration.neighbours.Where(n => n.sample != null).Take(k).ToList();

                                // find the max distance of the neighbours
                                double maxDistance = kNeighbours.Last().distance;

                                // iterate aggregation types
                                foreach (KNNConfiguration.Aggregation aggregation in knnConfiguration.aggregations)
                                {
                                    // iterate weight exponents
                                    for (double weightExponent = knnConfiguration.weightExponentMin; weightExponent <= knnConfiguration.weightExponentMax; weightExponent += knnConfiguration.weightExponentStep)
                                    {
                                        // create the result and weight sum
                                        double[] result = new double[dataset.outputLength];
                                        double weightSum = 0f;

                                        // iterate through neighbours
                                        foreach (Neighbour kNeighbour in kNeighbours)
                                        {
                                            // if the neighbour has a null sample, its too high a k value
                                            if (kNeighbour.sample == null)
                                            {
                                                throw new Exception("K value too high for KNN, not enough neighbours to consider");
                                            }

                                            // calculate the weight
                                            double weight;

                                            switch (aggregation)
                                            {
                                                case KNNConfiguration.Aggregation.Flat:
                                                    weight = 1.0;
                                                    break;
                                                case KNNConfiguration.Aggregation.InverseNormal:
                                                    weight = maxDistance / (kNeighbour.distance + KNNConfiguration.EPSILON);
                                                    break;
                                                case KNNConfiguration.Aggregation.Reciprocal:
                                                    weight = 1.0 / (kNeighbour.distance + KNNConfiguration.EPSILON);
                                                    break;
                                                default:
                                                    throw new Exception("Unknown aggregation type");
                                            }

                                            // we then exponentiate the weight to shape the falloff of neighbour influence (more exponent means closer neighbours matter more)
                                            weight = Math.Pow(weight, weightExponent);

                                            // add to weight sum
                                            weightSum += weight;

                                            // weight in the neighbours influenece on the result
                                            for (int i = 0; i < result.Length; i++)
                                            {
                                                result[i] += kNeighbour.sample.output[i] * weight;
                                            }
                                        }

                                        // we then take the weighted average of the accumulated neighbour results
                                        for (int i = 0; i < result.Length; i++)
                                        {
                                            result[i] /= weightSum;
                                        }

                                        // get the argmax
                                        int resultArgmax = Utility.Argmax(result);

                                        // if correct
                                        if (resultArgmax == correctArgmax)
                                        {
                                            scores.Tally(k, distanceBias, distanceExponent, distanceRoot, weightExponent, aggregation, set.isTrain);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // dump scores
        Console.WriteLine();
        scores.Dump($"./scores-{dataset.name}.csv");
    }
}