public static class KNN
{
    public static List<double[]> Run(KNNConfiguration knnConfiguration, Sample[] train, double[] testInput, int ignoreTrainIndex = -1)
    {
        // parallel compute distances to each neighbour
        Parallel.For(0, train.Length, trainIndex =>
        {
            // get the train sample at index
            Sample trainSample = train[trainIndex];

            // get the neighbour at index
            Neighbour neighbour = knnConfiguration.neighbours[trainIndex];

            // if this is the ignored sample (for leave one out self-optimization)
            if (trainIndex == ignoreTrainIndex)
            {
                // set null
                neighbour.index = -1;
                neighbour.sample = null;
                neighbour.distance = double.MaxValue;
                return;
            }

            // get the input for this train sample
            double[] trainInput = train[trainIndex].input;

            // compute the distance between the test and train inputs
            double distance = 0f;
            for (int inputIndex = 0; inputIndex < trainInput.Length; inputIndex++)
            {
                // we start with a bias distance of 1 between all inputs (creates a leverage for exponents and weights, even for identical inputs)
                double inputDistance = 1.0;

                // next we add the absolute difference between the inputs (absolute so its always positive, and can be raised to any power)
                inputDistance += Math.Abs(trainInput[inputIndex] - testInput[inputIndex]);

                // then we raise the input distance to the distance exponent (allowing for an emphasis on closer or further distances)
                inputDistance = Math.Pow(inputDistance, knnConfiguration.distanceExponent);

                // finally we multiply the input distance by the input weight (allowing for some inputs to be more important than others)
                inputDistance *= knnConfiguration.inputWeights[inputIndex];

                // this is added to the total distance
                distance += inputDistance;
            }

            // first we take the specified root of the distance (this is different than the distance exponent, to allow a different compression of the distance)
            distance = Math.Pow(distance, 1f / knnConfiguration.distanceRoot);

            // then finally we multiply by the distance weight for this train sample (this allows demotion or promotion of certain samples)
            distance *= knnConfiguration.distanceWeights[trainIndex];

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
            List<Neighbour> kNeighbours = knnConfiguration.neighbours.Take(k).ToList();

            // find the max distance of the neighbours
            double maxDistance = kNeighbours.Last().distance;

            // create the result and weight sum
            double[] result = new double[train[0].output.Length];
            double weightSum = 0f;

            // iterate through neighbours
            foreach (Neighbour kNeighbour in kNeighbours)
            {
                // if the neighbour has a null sample, its too high a k value
                if (kNeighbour.sample == null)
                {
                    throw new Exception("K value too high for KNN, not enough neighbours to consider");
                }

                // calculate the weight of this neighbour, the max distance will be 1, and closer distances will be relatively higher than 1
                double weight = maxDistance / kNeighbour.distance;

                // we then exponentiate the weight to shape the falloff of neighbour influence (more exponent means closer neighbours matter more)
                weight = Math.Pow(weight, knnConfiguration.weightExponent);

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

            // store the result
            results.Add(result);
        }

        // return results
        return results;
    }

    public static void Score(Dataset dataset, KNNConfiguration knnConfiguration, ref double[] kArgmaxErrorTrain, ref double[] kAbsoluteErrorTrain, ref double[] kArgmaxErrorTest, ref double[] kAbsoluteErrorTest, ref int[] kTrainCorrects, ref int[] kTestCorrects)
    {
        // zero results
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            kArgmaxErrorTrain[k] = 0;
            kAbsoluteErrorTrain[k] = 0f;
            kArgmaxErrorTest[k] = 0;
            kAbsoluteErrorTest[k] = 0f;
            kTrainCorrects[k] = 0;
            kTestCorrects[k] = 0;
        }

        // iterate through train samples
        for (int trainIndex = 0; trainIndex < dataset.train.Length; trainIndex++)
        {
            // get the train sample
            Sample trainSample = dataset.train[trainIndex];

            // run the KNN
            List<double[]> predictions = Run(knnConfiguration, dataset.train, trainSample.input, trainIndex);

            // get the actual output
            double[] actual = trainSample.output;

            // get the actual argmax
            int actualArgmax = dataset.trainArgmax[trainIndex];

            // iterate through k values
            for (int k = 1; k <= knnConfiguration.maxK; k++)
            {
                // get the prediction
                double[] kPrediction = predictions[k - 1];

                // get the argmax prediction
                int kArgmaxPrediction = Utility.Argmax(kPrediction);

                // if the prediction is incorrect
                if (kArgmaxPrediction != actualArgmax)
                {
                    // increment incorrect count
                    kArgmaxErrorTrain[k - 1] += 1;
                }
                else
                {
                    // increment correct count
                    kTrainCorrects[k - 1]++;
                }

                // iterate through outputs
                for (int i = 0; i < kPrediction.Length; i++)
                {
                    // add the absolute error
                    kAbsoluteErrorTrain[k - 1] += Math.Abs(kPrediction[i] - actual[i]);
                }
            }
        }

        // calculate argmax errors
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            kArgmaxErrorTrain[k] = kArgmaxErrorTrain[k] / (double)dataset.train.Length;
        }

        // iterate through test samples
        for (int testIndex = 0; testIndex < dataset.test.Length; testIndex++)
        {
            // get the test sample
            Sample testSample = dataset.test[testIndex];

            // run the KNN
            List<double[]> predictions = Run(knnConfiguration, dataset.train, testSample.input, -1);

            // get the actual output
            double[] actual = testSample.output;

            // get the actual argmax
            int actualArgmax = dataset.testArgmax[testIndex];

            // iterate through k values
            for (int k = 1; k <= knnConfiguration.maxK; k++)
            {
                // get the prediction
                double[] kPrediction = predictions[k - 1];

                // get the argmax prediction
                int kArgmaxPrediction = Utility.Argmax(kPrediction);

                // if the prediction is incorrect
                if (kArgmaxPrediction != actualArgmax)
                {
                    // increment incorrect count
                    kArgmaxErrorTest[k - 1]++;
                }
                else
                {
                    // increment correct count
                    kTestCorrects[k - 1]++;
                }

                // iterate through outputs
                for (int i = 0; i < kPrediction.Length; i++)
                {
                    // add the absolute error
                    kAbsoluteErrorTest[k - 1] += Math.Abs(kPrediction[i] - actual[i]);
                }
            }
        }

        // calculate argmax errors
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            kArgmaxErrorTest[k] = kArgmaxErrorTest[k] / (double)dataset.test.Length;
        }
    }
}