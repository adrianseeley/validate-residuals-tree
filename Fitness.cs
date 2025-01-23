public class Fitness
{
    public long epoch;
    public double argmaxCorrectsAverageTrain;
    public double absoluteErrorsAverageTrain;
    public double argmaxCorrectsAverageTest;
    public double absoluteErrorsAverageTest;
    public double argmaxCorrectsBestTrain;
    public double absoluteErrorsBestTrain;
    public double argmaxCorrectsBestTest;
    public double absoluteErrorsBestTest;
    public int[] kArgmaxCorrectsTrain;
    public double[] kAbsoluteErrorsTrain;
    public int[] kArgmaxCorrectsTest;
    public double[] kAbsoluteErrorsTest;
    public TextWriter log;

    public Fitness(Dataset dataset, KNNConfiguration knnConfiguration)
    {
        this.epoch = 0;
        this.argmaxCorrectsAverageTrain = double.NaN;
        this.absoluteErrorsAverageTrain = double.NaN;
        this.argmaxCorrectsAverageTest = double.NaN;
        this.absoluteErrorsAverageTest = double.NaN;
        this.argmaxCorrectsBestTrain = double.NaN;
        this.absoluteErrorsBestTrain = double.NaN;
        this.argmaxCorrectsBestTest = double.NaN;
        this.absoluteErrorsBestTest = double.NaN;
        this.kArgmaxCorrectsTrain = new int[knnConfiguration.maxK];
        this.kAbsoluteErrorsTrain = new double[knnConfiguration.maxK];
        this.kArgmaxCorrectsTest = new int[knnConfiguration.maxK];
        this.kAbsoluteErrorsTest = new double[knnConfiguration.maxK];
        this.log = new StreamWriter($"./{dataset.name}-log.csv");
        this.log.WriteLine("epoch,argmaxCorrectsAverageTrain,absoluteErrorsAverageTrain,argmaxCorrectsAverageTest,absoluteErrorsAverageTest,argmaxCorrectsBestTrain,absoluteErrorsBestTrain,argmaxCorrectsBestTest,absoluteErrorsBestTest");
        this.log.Flush();
    }

    public bool CheckImproved(Dataset dataset, KNNConfiguration knnConfiguration)
    {
        epoch++;
        KNN.Score(dataset, knnConfiguration, ref kArgmaxCorrectsTrain, ref kAbsoluteErrorsTrain, ref kArgmaxCorrectsTest, ref kAbsoluteErrorsTest);
        double argmaxCorrectsAverageTrain = kArgmaxCorrectsTrain.Average();
        double absoluteErrorsAverageTrain = kAbsoluteErrorsTrain.Average();
        double argmaxCorrectsAverageTest = kArgmaxCorrectsTest.Average();
        double absoluteErrorsAverageTest = kAbsoluteErrorsTest.Average();
        double argmaxCorrectsBestTrain = kArgmaxCorrectsTrain.Max();
        double absoluteErrorsBestTrain = kAbsoluteErrorsTrain.Min();
        double argmaxCorrectsBestTest = kArgmaxCorrectsTest.Max();
        double absoluteErrorsBestTest = kAbsoluteErrorsTest.Min();

        // improvement is defined by a decrease in the absolute errors average train
        bool improved = false;
        if (double.IsNaN(this.absoluteErrorsAverageTrain) || absoluteErrorsAverageTrain < this.absoluteErrorsAverageTrain)
        {
            this.argmaxCorrectsAverageTrain = argmaxCorrectsAverageTrain;
            this.absoluteErrorsAverageTrain = absoluteErrorsAverageTrain;
            this.argmaxCorrectsAverageTest = argmaxCorrectsAverageTest;
            this.absoluteErrorsAverageTest = absoluteErrorsAverageTest;
            this.argmaxCorrectsBestTrain = argmaxCorrectsBestTrain;
            this.absoluteErrorsBestTrain = absoluteErrorsBestTrain;
            this.argmaxCorrectsBestTest = argmaxCorrectsBestTest;
            this.absoluteErrorsBestTest = absoluteErrorsBestTest;
            improved = true;
        }
        log.WriteLine($"{epoch},{this.argmaxCorrectsAverageTrain},{this.absoluteErrorsAverageTrain},{this.argmaxCorrectsAverageTest},{this.absoluteErrorsAverageTest},{this.argmaxCorrectsBestTrain},{this.absoluteErrorsBestTrain},{this.argmaxCorrectsBestTest},{this.absoluteErrorsBestTest}");
        Console.Write($"\rEpoch: {epoch}, Error: {this.absoluteErrorsAverageTrain}");
        return improved;
    }

    public void FinalizeLog()
    {
        Console.WriteLine();
        log.Close();
    }
}