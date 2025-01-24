public class NeighbourPool
{
    public List<NeighbourPoolEntry> entries;
    public int outputLength;

    public NeighbourPool(Dataset dataset)
    {
        entries = new List<NeighbourPoolEntry>();
        outputLength = dataset.outputLength;

        TextWriter log = new StreamWriter("./log.csv");
        log.WriteLine("entryCount,argmaxErrorTrain,argmaxErrorTest,absoluteErrorTrain,absoluteErrorTest");
        log.Flush();

        Random random = new Random();
        bool[] closerA = new bool[dataset.train.Length];
        for (; ;)
        {
            double[] randomAPosition = new double[dataset.inputLength];
            double[] randomBPosition = new double[dataset.inputLength];
            for (int i = 0; i < dataset.inputLength; i++)
            {
                randomAPosition[i] = random.NextDouble();
                randomBPosition[i] = random.NextDouble();
            }

            int aCount = 0;
            int bCount = 0;
            Parallel.For(0, dataset.train.Length, trainIndex =>
            {
                double aDistance = Utility.EuclideanDistance(dataset.train[trainIndex].input, randomAPosition);
                double bDistance = Utility.EuclideanDistance(dataset.train[trainIndex].input, randomBPosition);
                if (aDistance <= bDistance)
                {
                    closerA[trainIndex] = true;
                    Interlocked.Increment(ref aCount);
                }
                else
                {
                    closerA[trainIndex] = false;
                    Interlocked.Increment(ref bCount);
                }
            });

            if (aCount == 0 || bCount == 0)
            {
                continue;
            }

            double[] aValue = new double[outputLength];
            double[] bValue = new double[outputLength];
            for (int trainIndex = 0; trainIndex < dataset.train.Length; trainIndex++)
            {
                double[] addInto = closerA[trainIndex] ? aValue : bValue;
                for (int i = 0; i < outputLength; i++)
                {
                    addInto[i] += dataset.train[trainIndex].output[i];
                }
            }
            for (int i = 0; i < outputLength; i++)
            {
                aValue[i] /= aCount;
                bValue[i] /= bCount;
            }
            entries.Add(new NeighbourPoolEntry(randomAPosition, randomBPosition, aValue, bValue));

            if (entries.Count % 1000 == 0)
            {
                (double argmaxErrorTrain, double argmaxErrorTest, double absoluteErrorTrain, double absoluteErrorTest) = Test(dataset);
                log.WriteLine($"{entries.Count},{argmaxErrorTrain},{argmaxErrorTest},{absoluteErrorTrain},{absoluteErrorTest}");
                log.Flush();
                Console.WriteLine($"count: {entries.Count}, argmaxErrorTrain: {argmaxErrorTrain}, argmaxErrorTest: {argmaxErrorTest}, absoluteErrorTrain: {absoluteErrorTrain}, absoluteErrorTest: {absoluteErrorTest}");
            }
        }
    }

    public (double argmaxErrorTrain, double argmaxErrorTest, double absoluteErrorTrain, double absoluteErrorTest) Test(Dataset dataset)
    {
        double argmaxErrorTrain = 0;
        double argmaxErrorTest = 0;
        double absoluteErrorTrain = 0;
        double absoluteErrorTest = 0;
        foreach (Sample sample in dataset.train)
        {
            double[] prediction = Predict(sample.input);
            if (Utility.Argmax(prediction) != Utility.Argmax(sample.output))
            {
                argmaxErrorTrain++;
            }
            absoluteErrorTrain += Utility.AbsoluteError(sample.output, prediction);
        }
        foreach (Sample sample in dataset.test)
        {
            double[] prediction = Predict(sample.input);
            if (Utility.Argmax(prediction) != Utility.Argmax(sample.output))
            {
                argmaxErrorTest++;
            }
            absoluteErrorTest += Utility.AbsoluteError(sample.output, prediction);
        }
        argmaxErrorTrain /= dataset.train.Length;
        argmaxErrorTest /= dataset.test.Length;
        return (argmaxErrorTrain, argmaxErrorTest, absoluteErrorTrain, absoluteErrorTest);
    }

    public double[] Predict(double[] input)
    {
        bool[] closerA = new bool[entries.Count];
        Parallel.For(0, entries.Count, entryIndex =>
        {
            double aDistance = Utility.EuclideanDistance(input, entries[entryIndex].aPosition);
            double bDistance = Utility.EuclideanDistance(input, entries[entryIndex].bPosition);
            if (aDistance <= bDistance)
            {
                closerA[entryIndex] = true;
            }
            else
            {
                closerA[entryIndex] = false;
            }
        });
        double[] prediction = new double[outputLength];
        for (int entryIndex = 0; entryIndex < entries.Count; entryIndex++)
        {
            double[] entryValue = closerA[entryIndex] ? entries[entryIndex].aValue : entries[entryIndex].bValue;
            for (int i = 0; i < outputLength; i++)
            {
                prediction[i] += entryValue[i];
            }
        }
        for (int i = 0; i < outputLength; i++)
        {
            prediction[i] /= entries.Count;
        }
        return prediction;
    }
}

public class NeighbourPoolEntry
{
    public double[] aPosition;
    public double[] bPosition;
    public double[] aValue;
    public double[] bValue;

    public NeighbourPoolEntry(double[] aPosition, double[] bPosition, double[] aValue, double[] bValue)
    {
        this.aPosition = aPosition;
        this.bPosition = bPosition;
        this.aValue = aValue;
        this.bValue = bValue;
    }
}