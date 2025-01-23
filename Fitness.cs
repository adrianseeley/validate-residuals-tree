public class Fitness
{
    public Dataset dataset;
    public KNNConfiguration knnConfiguration;
    public long epoch;
    public double argmaxErrorAverageTrain;
    public double absoluteErrorAverageTrain;
    public double argmaxErrorAverageTest;
    public double absoluteErrorAverageTest;
    public double argmaxErrorBestTrain;
    public double absoluteErrorBestTrain;
    public double argmaxErrorBestTest;
    public double absoluteErrorBestTest;
    public double[] kArgmaxErrorTrain;
    public double[] kAbsoluteErrorTrain;
    public double[] kArgmaxErrorTest;
    public double[] kAbsoluteErrorTest;
    public TextWriter log;

    public Fitness(Dataset dataset, KNNConfiguration knnConfiguration)
    {
        this.dataset = dataset;
        this.knnConfiguration = knnConfiguration;
        this.epoch = 0;
        this.argmaxErrorAverageTrain = double.NaN;
        this.absoluteErrorAverageTrain = double.NaN;
        this.argmaxErrorAverageTest = double.NaN;
        this.absoluteErrorAverageTest = double.NaN;
        this.argmaxErrorBestTrain = double.NaN;
        this.absoluteErrorBestTrain = double.NaN;
        this.argmaxErrorBestTest = double.NaN;
        this.absoluteErrorBestTest = double.NaN;
        this.kArgmaxErrorTrain = new double[knnConfiguration.maxK];
        this.kAbsoluteErrorTrain = new double[knnConfiguration.maxK];
        this.kArgmaxErrorTest = new double[knnConfiguration.maxK];
        this.kAbsoluteErrorTest = new double[knnConfiguration.maxK];
        this.log = new StreamWriter($"./{dataset.name}-log.csv");
        this.log.WriteLine("epoch,argmaxErrorAverageTrain,absoluteErrorAverageTrain,argmaxErrorAverageTest,absoluteErrorAverageTest,argmaxErrorBestTrain,absoluteErrorBestTrain,argmaxErrorBestTest,absoluteErrorBestTest");
        this.log.Flush();
    }

    public bool CheckImproved()
    {
        epoch++;
        KNN.Score(dataset, knnConfiguration, ref kArgmaxErrorTrain, ref kAbsoluteErrorTrain, ref kArgmaxErrorTest, ref kAbsoluteErrorTest);
        double argmaxErrorAverageTrain = kArgmaxErrorTrain.Average();
        double absoluteErrorAverageTrain = kAbsoluteErrorTrain.Average();
        double argmaxErrorAverageTest = kArgmaxErrorTest.Average();
        double absoluteErrorAverageTest = kAbsoluteErrorTest.Average();
        double argmaxErrorBestTrain = kArgmaxErrorTrain.Min();
        double absoluteErrorBestTrain = kAbsoluteErrorTrain.Min();
        double argmaxErrorBestTest = kArgmaxErrorTest.Min();
        double absoluteErrorBestTest = kAbsoluteErrorTest.Min();

        // improvement is defined by a decrease in the absolute error average train
        bool improved = false;
        if (double.IsNaN(this.absoluteErrorAverageTrain) || absoluteErrorAverageTrain < this.absoluteErrorAverageTrain)
        {
            this.argmaxErrorAverageTrain = argmaxErrorAverageTrain;
            this.absoluteErrorAverageTrain = absoluteErrorAverageTrain;
            this.argmaxErrorAverageTest = argmaxErrorAverageTest;
            this.absoluteErrorAverageTest = absoluteErrorAverageTest;
            this.argmaxErrorBestTrain = argmaxErrorBestTrain;
            this.absoluteErrorBestTrain = absoluteErrorBestTrain;
            this.argmaxErrorBestTest = argmaxErrorBestTest;
            this.absoluteErrorBestTest = absoluteErrorBestTest;
            improved = true;
        }
        log.WriteLine($"{epoch},{argmaxErrorAverageTrain},{absoluteErrorAverageTrain},{argmaxErrorAverageTest},{absoluteErrorAverageTest},{argmaxErrorBestTrain},{absoluteErrorBestTrain},{argmaxErrorBestTest},{absoluteErrorBestTest}");
        log.Flush();
        Console.Write($"\rEpoch: {epoch}, Error: {this.absoluteErrorAverageTrain}");
        return improved;
    }

    public void FinalizeLog()
    {
        Console.WriteLine();
        log.Close();

        // create predictions log for each k value train and test
        List<TextWriter> trainPredictionsLog = new List<TextWriter>();
        List<TextWriter> testPredictionsLog = new List<TextWriter>();
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            trainPredictionsLog.Add(new StreamWriter($"./{dataset.name}-train-predictions-k{k + 1}.csv"));
            testPredictionsLog.Add(new StreamWriter($"./{dataset.name}-test-predictions-k{k + 1}.csv"));
            trainPredictionsLog[k].WriteLine("actual,prediction,argmaxActual,argmaxPrediction,absoluteError");
            testPredictionsLog[k].WriteLine("actual,prediction,argmaxActual,argmaxPrediction,absoluteError");
        }

        // iterate train samples
        int[] kTrainCorrect = new int[knnConfiguration.maxK];
        for (int trainIndex = 0; trainIndex < dataset.train.Length; trainIndex++)
        {
            // get the train sample
            Sample trainSample = dataset.train[trainIndex];

            // predict the train sample
            List<double[]> predictions = KNN.Run(knnConfiguration, dataset.train, trainSample.input, trainIndex);

            // get argmax actual
            int argmaxActual = dataset.trainArgmax[trainIndex];

            // iterate k values
            for (int k = 0; k < knnConfiguration.maxK; k++)
            {
                // get argmax prediction
                int argmaxPrediction = Utility.Argmax(predictions[k]);

                // check if answer was correct
                if (argmaxActual == argmaxPrediction)
                {
                    kTrainCorrect[k]++;
                }

                // get absolute error
                double absoluteError = Utility.AbsoluteError(dataset.train[trainIndex].output, predictions[k]);

                // write to log
                trainPredictionsLog[k].WriteLine($"[{string.Join(", ", trainSample.output)}],[{string.Join(", ", predictions[k])}],{argmaxActual},{argmaxPrediction},{absoluteError}");
            }
        }

        // iterate test samples
        int[] kTestCorrect = new int[knnConfiguration.maxK];
        for (int testIndex = 0; testIndex < dataset.test.Length; testIndex++)
        {
            // predict the test sample
            List<double[]> predictions = KNN.Run(knnConfiguration, dataset.train, dataset.test[testIndex].input, -1);
            
            // get argmax actual
            int argmaxActual = dataset.testArgmax[testIndex];
            
            // iterate k values
            for (int k = 0; k < knnConfiguration.maxK; k++)
            {
                // get argmax prediction
                int argmaxPrediction = Utility.Argmax(predictions[k]);

                // check if answer was correct
                if (argmaxActual == argmaxPrediction)
                {
                    kTestCorrect[k]++;
                }

                // get absolute error
                double absoluteError = Utility.AbsoluteError(dataset.test[testIndex].output, predictions[k]);

                // write to log
                testPredictionsLog[k].WriteLine($"[{string.Join(", ", dataset.test[testIndex].output)}],[{string.Join(", ", predictions[k])}],{argmaxActual},{argmaxPrediction},{absoluteError}");
            }
        }

        // close predictions log
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            trainPredictionsLog[k].Close();
            testPredictionsLog[k].Close();
        }

        // create final log
        TextWriter finalLog = new StreamWriter($"./{dataset.name}-final-log.csv");
        finalLog.WriteLine("k,argmaxErrorTrain,absoluteErrorTrain,argmaxErrorTest,absoluteErrorTest,trainCorrect,trainTotal,testCorrect,testTotal");

        // get final score
        KNN.Score(dataset, knnConfiguration, ref kArgmaxErrorTrain, ref kAbsoluteErrorTrain, ref kArgmaxErrorTest, ref kAbsoluteErrorTest);

        // write results
        for (int k = 0; k < knnConfiguration.maxK; k++)
        {
            finalLog.WriteLine($"{k + 1},{kArgmaxErrorTrain[k]},{kAbsoluteErrorTrain[k]},{kArgmaxErrorTest[k]},{kAbsoluteErrorTest[k]},{kTrainCorrect[k]},{dataset.train.Length},{kTestCorrect[k]},{dataset.test.Length}");
            Console.WriteLine($"K: {k + 1}, Train(Argmax): {kArgmaxErrorTrain[k]}, Train(Absolute): {kAbsoluteErrorTrain[k]}, Test(Argmax): {kArgmaxErrorTest[k]}, Test(Absolute): {kAbsoluteErrorTest[k]}, Train Correct: {kTrainCorrect[k]}/{dataset.train.Length}, Test Correct: {kTestCorrect[k]}/{dataset.test.Length}");
        }

        // close final log
        finalLog.Close();
    }
}