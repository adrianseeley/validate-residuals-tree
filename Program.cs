public class Program
{
    public static void Main(string[] args)
    {
        int maxK = 10;
        double nudge = 0.1;
        List<Dataset> datasets = new List<Dataset>();

        foreach (Dataset dataset in datasets)
        {
            // create the knn configuration
            KNNConfiguration knnConfiguration = new KNNConfiguration(dataset, maxK);
            
            // create the fitness tracker
            Fitness fitness = new Fitness(dataset, knnConfiguration);
            
            // set the starting fitness
            fitness.CheckImproved(dataset, knnConfiguration);

            // loop until improvements stop
            bool improved = true;
            while (improved)
            {
                improved = false;


                // try distance exponent down
                knnConfiguration.distanceExponent -= nudge;
                if (fitness.CheckImproved(dataset, knnConfiguration))
                {
                    improved = true;
                }
                else
                {
                    // try distance exponent up
                    knnConfiguration.distanceExponent += 2 * nudge;
                    if (fitness.CheckImproved(dataset, knnConfiguration))
                    {
                        improved = true;
                    }
                    else
                    {
                        // reset to original value
                        knnConfiguration.distanceExponent -= nudge;
                    }
                }

                // try distance root down
                knnConfiguration.distanceRoot -= nudge;
                if (fitness.CheckImproved(dataset, knnConfiguration))
                {
                    improved = true;
                }
                else
                {
                    // try distance root up
                    knnConfiguration.distanceRoot += 2 * nudge;
                    if (fitness.CheckImproved(dataset, knnConfiguration))
                    {
                        improved = true;
                    }
                    else
                    {
                        // reset to original value
                        knnConfiguration.distanceRoot -= nudge;
                    }
                }

                // try weight exponent down
                knnConfiguration.weightExponent -= nudge;
                if (fitness.CheckImproved(dataset, knnConfiguration))
                {
                    improved = true;
                }
                else
                {
                    // try weight exponent up
                    knnConfiguration.weightExponent += 2 * nudge;
                    if (fitness.CheckImproved(dataset, knnConfiguration))
                    {
                        improved = true;
                    }
                    else
                    {
                        // reset to original value
                        knnConfiguration.weightExponent -= nudge;
                    }
                }

                // iterate input weights
                for (int inputIndex = 0; inputIndex < knnConfiguration.inputWeights.Length; inputIndex++)
                {
                    // try input weight down
                    knnConfiguration.inputWeights[inputIndex] -= nudge;
                    if (fitness.CheckImproved(dataset, knnConfiguration))
                    {
                        improved = true;
                    }
                    else
                    {
                        // try input weight up
                        knnConfiguration.inputWeights[inputIndex] += 2 * nudge;
                        if (fitness.CheckImproved(dataset, knnConfiguration))
                        {
                            improved = true;
                        }
                        else
                        {
                            // reset to original value
                            knnConfiguration.inputWeights[inputIndex] -= nudge;
                        }
                    }
                }

                // iterate distance weights
                for (int distanceIndex = 0; distanceIndex < knnConfiguration.distanceWeights.Length; distanceIndex++)
                {
                    // try distance weight down
                    knnConfiguration.distanceWeights[distanceIndex] -= nudge;
                    if (fitness.CheckImproved(dataset, knnConfiguration))
                    {
                        improved = true;
                    }
                    else
                    {
                        // try distance weight up
                        knnConfiguration.distanceWeights[distanceIndex] += 2 * nudge;
                        if (fitness.CheckImproved(dataset, knnConfiguration))
                        {
                            improved = true;
                        }
                        else
                        {
                            // reset to original value
                            knnConfiguration.distanceWeights[distanceIndex] -= nudge;
                        }
                    }
                }
            }

            // improvement complete, finalize log
            fitness.FinalizeLog();
        }
    }
}