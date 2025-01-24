public class Program
{
    public static void Main(string[] args)
    {
        int maxK = 10;
        List<Dataset> datasets = new List<Dataset>()
        {
            //Data.IRIS("./data/IRIS/iris.data"),
            Data.MNIST("d:/data/mnist_train.csv", "d:/data/mnist_test.csv", 1000),
        };

        double[] kArgmaxErrorTrain = new double[maxK];
        double[] kAbsoluteErrorTrain = new double[maxK];
        double[] kArgmaxErrorTest = new double[maxK];
        double[] kAbsoluteErrorTest = new double[maxK];
        int[] kTrainCorrects = new int[maxK];
        int[] kTestCorrects = new int[maxK];

        foreach (Dataset dataset in datasets)
        {
            // create the knn configuration
            KNNConfiguration knnConfiguration = new KNNConfiguration(dataset, maxK);
            
            // create log
            TextWriter log = new StreamWriter($"./log-{dataset.name}.csv");
            log.WriteLine("aggregation,k,inputDistanceBias,distanceExponent,distanceRoot,weightExponent,argmaxErrorTrain,absoluteErrorTrain,argmaxErrorTest,absoluteErrorTest,trainCorrect,testCorrect,trainTotal,testTotal");
            log.Flush();

            // grid search
            double baseInputDistanceBias = 0.0;
            double inputDistanceBiasMax = 7.5;
            double inputDistanceBiasStep = 0.25;
            double baseExponent = 1.0;
            double maxExponent = 30.0;
            double exponentStep = 1;
            KNNConfiguration.Aggregation[] aggregations = [KNNConfiguration.Aggregation.Flat, KNNConfiguration.Aggregation.InverseNormal, KNNConfiguration.Aggregation.Reciprocal];
            for (double inputDistanceBias = baseInputDistanceBias; inputDistanceBias <= inputDistanceBiasMax; inputDistanceBias += inputDistanceBiasStep)
            {
                for (double distanceExponent = baseExponent; distanceExponent <= maxExponent; distanceExponent += exponentStep)
                {
                    for (double distanceRoot = baseExponent; distanceRoot <= maxExponent; distanceRoot += exponentStep)
                    {
                        for (double weightExponent = baseExponent; weightExponent <= maxExponent; weightExponent += exponentStep)
                        {
                            foreach (KNNConfiguration.Aggregation aggregation in aggregations)
                            {
                                // set the configuration
                                knnConfiguration.inputDistanceBias = inputDistanceBias;
                                knnConfiguration.distanceExponent = distanceExponent;
                                knnConfiguration.distanceRoot = distanceRoot;
                                knnConfiguration.weightExponent = weightExponent;
                                knnConfiguration.aggregation = aggregation;

                                // score the configuration
                                KNN.Score(dataset, knnConfiguration, ref kArgmaxErrorTrain, ref kAbsoluteErrorTrain, ref kArgmaxErrorTest, ref kAbsoluteErrorTest, ref kTrainCorrects, ref kTestCorrects);

                                // log resutls
                                for (int k = 0; k < maxK; k++)
                                {
                                    log.WriteLine($"{aggregation},{k + 1},{inputDistanceBias},{distanceExponent},{distanceRoot},{weightExponent},{kArgmaxErrorTrain[k]},{kAbsoluteErrorTrain[k]},{kArgmaxErrorTest[k]},{kAbsoluteErrorTest[k]},{kTrainCorrects[k]},{kTestCorrects[k]},{dataset.train.Length},{dataset.test.Length}");
                                    Console.WriteLine($"{dataset.name} aggregation: {aggregation} k: {k + 1} inputDistanceBias: {inputDistanceBias} distanceExponent: {distanceExponent} distanceRoot: {distanceRoot} weightExponent: {weightExponent} argmaxErrorTrain: {kArgmaxErrorTrain[k]} absoluteErrorTrain: {kAbsoluteErrorTrain[k]} argmaxErrorTest: {kArgmaxErrorTest[k]} absoluteErrorTest: {kAbsoluteErrorTest[k]} trainCorrect: {kTrainCorrects[k]} testCorrect: {kTestCorrects[k]} trainTotal: {dataset.train.Length} testTotal: {dataset.test.Length}");
                                }
                                log.Flush();
                            }
                        }
                    }
                }
            }

            // finialize log
            log.Close();
        }
    }
}