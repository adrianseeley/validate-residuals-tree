public class Program
{
    public static void Main(string[] args)
    {
        int maxK = 10;
        double nudge = 0.1;
        double requiredPassImprovement = 0.0000001;
        List<Dataset> datasets = new List<Dataset>()
        {
            //Data.IRIS("./data/IRIS/iris.data"),
            Data.MNIST("d:/data/mnist_train.csv", "d:/data/mnist_test.csv", 1000),
        };


        foreach (Dataset dataset in datasets)
        {
            // create the knn configuration
            KNNConfiguration knnConfiguration = new KNNConfiguration(dataset, maxK);
            
            // create the fitness tracker
            Fitness fitness = new Fitness(dataset, knnConfiguration);
            
            // set the starting fitness
            fitness.CheckImproved();

            // loop until required pass improvement isnt met
            bool requirementMet = true;
            while (requirementMet)
            {
                // start the starting fitness
                double passStartFitness = fitness.absoluteErrorAverageTrain;

                // try distance exponent down
                knnConfiguration.distanceExponent -= nudge;
                if (!fitness.CheckImproved())
                {
                    // try distance exponent up
                    knnConfiguration.distanceExponent += 2 * nudge;
                    if (!fitness.CheckImproved())
                    {
                        // reset to original value
                        knnConfiguration.distanceExponent -= nudge;
                    }
                }

                // try distance root down
                knnConfiguration.distanceRoot -= nudge;
                if (!fitness.CheckImproved())
                {
                    // try distance root up
                    knnConfiguration.distanceRoot += 2 * nudge;
                    if (!fitness.CheckImproved())
                    {
                        // reset to original value
                        knnConfiguration.distanceRoot -= nudge;
                    }
                }

                // try weight exponent down
                knnConfiguration.weightExponent -= nudge;
                if (!fitness.CheckImproved())
                {
                    // try weight exponent up
                    knnConfiguration.weightExponent += 2 * nudge;
                    if (!fitness.CheckImproved())
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
                    if (!fitness.CheckImproved())
                    {
                        // try input weight up
                        knnConfiguration.inputWeights[inputIndex] += 2 * nudge;
                        if (!fitness.CheckImproved())
                        {
                            // reset to original value
                            knnConfiguration.inputWeights[inputIndex] -= nudge;
                        }
                    }
                }
                /*
                // iterate distance weights
                for (int distanceIndex = 0; distanceIndex < knnConfiguration.distanceWeights.Length; distanceIndex++)
                {
                    // try distance weight down
                    knnConfiguration.distanceWeights[distanceIndex] -= nudge;
                    if (!fitness.CheckImproved())
                    {
                        // try distance weight up
                        knnConfiguration.distanceWeights[distanceIndex] += 2 * nudge;
                        if (!fitness.CheckImproved())
                        {
                            // reset to original value
                            knnConfiguration.distanceWeights[distanceIndex] -= nudge;
                        }
                    }
                }
                */

                // check the pass improvement
                double passEndFitness = fitness.absoluteErrorAverageTrain;
                double passImprovement = passStartFitness - passEndFitness;
                requirementMet = passImprovement > requiredPassImprovement;

                Console.WriteLine($"\nPass Improvement: {passImprovement}");
            }

            // improvement complete, finalize log
            fitness.FinalizeLog();
        }
    }
}