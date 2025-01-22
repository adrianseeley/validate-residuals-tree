public class Sample
{
    public double[] input;
    public double[] output;

    public Sample(double[] input, double[] output)
    {
        this.input = input;
        this.output = output;
    }
}

public abstract class Tree
{
    public float[] prediction;
    public float error;
    public int currentDepth;
    public int maxDepth;
    public int minLeafSize;
    public int? splitInputIndex;
    public float? splitInputValue;
    public Tree? left;
    public Tree? right;

    public Tree(List<Sample> samples, int maxDepth, int minLeafSize, int currentDepth = 0)
    {
        this.prediction = DeterminePrediction(samples);
        this.error = DetermineError(prediction, samples);
        this.currentDepth = currentDepth;
        this.maxDepth = maxDepth;
        this.minLeafSize = minLeafSize;
        this.splitInputIndex = null;
        this.splitInputValue = null;
        this.left = null;
        this.right = null;
        if (samples.Count <= minLeafSize * 2 || currentDepth >= maxDepth)
        {
            return;
        }
        Split(samples);
    }

    public abstract void Split(List<Sample> samples);

    public float[] Predict(float[] input)
    {
        if (splitInputIndex == null || splitInputValue == null)
        {
            return prediction;
        }
        if (left == null || right == null)
        {
            throw new Exception("Invalid tree");
        }
        if (input[splitInputIndex.Value] < splitInputValue.Value)
        {
            return left.Predict(input);
        }
        else
        {
            return right.Predict(input);
        }
    }

    public float Prune(List<Sample> validationSamples)
    {
        float validationError = DetermineError(prediction, validationSamples);
        if (splitInputIndex == null || splitInputValue == null)
        {
            return validationError;
        }
        if (left == null || right == null)
        {
            throw new Exception("Invalid tree");
        }
        List<Sample> leftValidationSamples = new List<Sample>();
        List<Sample> rightValidationSamples = new List<Sample>();
        foreach (Sample validationSample in validationSamples)
        {
            if (validationSample.input[splitInputIndex.Value] < splitInputValue.Value)
            {
                leftValidationSamples.Add(validationSample);
            }
            else
            {
                rightValidationSamples.Add(validationSample);
            }
        }

        // if we dont use the whole split, prune
        if (leftValidationSamples.Count == 0 || rightValidationSamples.Count == 0)
        {
            splitInputIndex = null;
            splitInputValue = null;
            left = null;
            right = null;
            return validationError;
        }

        float leftValidationError = left.Prune(leftValidationSamples);
        float rightValidationError = right.Prune(rightValidationSamples);
        float leftWeight = (float)leftValidationSamples.Count / (float)validationSamples.Count;
        float rightWeight = (float)rightValidationSamples.Count / (float)validationSamples.Count;
        float jointValidationError = (leftWeight * leftValidationError) + (rightWeight * rightValidationError);

        // if error gets worse after split, prune
        if (jointValidationError >= validationError)
        {
            splitInputIndex = null;
            splitInputValue = null;
            left = null;
            right = null;
        }
        return validationError;
    }

    public static float[] DeterminePrediction(List<Sample> samples)
    {
        int outputCount = samples[0].output.Length;
        float[] prediction = new float[outputCount];
        foreach (Sample sample in samples)
        {
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                prediction[outputIndex] += sample.output[outputIndex];
            }
        }
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            prediction[outputIndex] /= (float)samples.Count;
        }
        return prediction;
    }

    public static float DetermineError(float[] prediction, List<Sample> samples)
    {
        float error = 0f;
        foreach (Sample sample in samples)
        {
            for (int outputIndex = 0; outputIndex < prediction.Length; outputIndex++)
            {
                error += MathF.Abs(sample.output[outputIndex] - prediction[outputIndex]);
            }
        }
        return error / (float)samples.Count;
    }
}

public class OptimalTree : Tree
{
    public OptimalTree(List<Sample> samples, int maxDepth, int minLeafSize, int currentDepth = 0)
        : base(samples, maxDepth, minLeafSize, currentDepth) { }

    public override void Split(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        int bestInputIndex = -1;
        float bestInputValue = float.NaN;
        float bestSplitError = error;
        List<Sample> leftSamples = new List<Sample>(samples.Count);
        List<Sample> rightSamples = new List<Sample>(samples.Count);
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            List<float> inputValues = samples.Select(sample => sample.input[inputIndex]).Distinct().OrderBy(value => value).ToList();
            for (int valueIndex = 0; valueIndex < inputValues.Count - 1; valueIndex++)
            {
                float inputValue = (inputValues[valueIndex] + inputValues[valueIndex + 1]) / 2;
                leftSamples.Clear();
                rightSamples.Clear();
                foreach (Sample sample in samples)
                {
                    if (sample.input[inputIndex] < inputValue)
                    {
                        leftSamples.Add(sample);
                    }
                    else
                    {
                        rightSamples.Add(sample);
                    }
                }
                if (leftSamples.Count < minLeafSize || rightSamples.Count < minLeafSize)
                {
                    continue;
                }
                float leftError = DetermineError(DeterminePrediction(leftSamples), leftSamples);
                float rightError = DetermineError(DeterminePrediction(rightSamples), rightSamples);
                float leftWeight = (float)leftSamples.Count / (float)samples.Count;
                float rightWeight = (float)rightSamples.Count / (float)samples.Count;
                float splitError = (leftWeight * leftError) + (rightWeight * rightError);
                if (float.IsNaN(bestSplitError) || splitError < bestSplitError)
                {
                    bestInputIndex = inputIndex;
                    bestInputValue = inputValue;
                    bestSplitError = splitError;
                }
            }
        }
        if (bestInputIndex == -1)
        {
            return;
        }
        splitInputIndex = bestInputIndex;
        splitInputValue = bestInputValue;
        leftSamples.Clear();
        rightSamples.Clear();
        foreach (Sample sample in samples)
        {
            if (sample.input[splitInputIndex.Value] < splitInputValue.Value)
            {
                leftSamples.Add(sample);
            }
            else
            {
                rightSamples.Add(sample);
            }
        }
        left = new OptimalTree(leftSamples, maxDepth, minLeafSize, currentDepth + 1);
        right = new OptimalTree(rightSamples, maxDepth, minLeafSize, currentDepth + 1);
    }
}

public class RandomTree : Tree
{
    public RandomTree(List<Sample> samples, int maxDepth, int minLeafSize, int currentDepth = 0)
        : base(samples, maxDepth, minLeafSize, currentDepth) { }

    public override void Split(List<Sample> samples)
    {
        Random random = new Random();
        int inputCount = samples[0].input.Length;
        List<Sample> leftSamples = new List<Sample>(samples.Count);
        List<Sample> rightSamples = new List<Sample>(samples.Count);

        List<int> inputIndices = Enumerable.Range(0, inputCount).OrderBy(i => random.NextSingle()).ToList();
        foreach (int inputIndex in inputIndices)
        {
            List<float> inputValues = samples.Select(sample => sample.input[inputIndex]).Distinct().OrderBy(i => random.NextSingle()).ToList();
            for (int inputValueIndex = 0; inputValueIndex < inputValues.Count - 1; inputValueIndex++)
            {
                float inputValue = (inputValues[inputValueIndex] + inputValues[inputValueIndex + 1]) / 2;
                leftSamples.Clear();
                rightSamples.Clear();
                foreach (Sample sample in samples)
                {
                    if (sample.input[inputIndex] < inputValue)
                    {
                        leftSamples.Add(sample);
                    }
                    else
                    {
                        rightSamples.Add(sample);
                    }
                }
                if (leftSamples.Count < minLeafSize || rightSamples.Count < minLeafSize)
                {
                    continue;
                }
                splitInputIndex = inputIndex;
                splitInputValue = inputValue;
                left = new RandomTree(leftSamples, maxDepth, minLeafSize, currentDepth + 1);
                right = new RandomTree(rightSamples, maxDepth, minLeafSize, currentDepth + 1);
                return;
            }
        }
    }
}

public class ResidualTrees
{
    public Random random;
    public List<Sample> residuals;
    public List<Tree> trees;
    public float learningRate;
    public float subsampleFraction;
    public int maxDepth;
    public int minLeafSize;
    public Type treeType;

    public ResidualTrees(List<Sample> samples, float learningRate, float subsampleFraction, int maxDepth, int minLeafSize, Type treeType)
    {
        this.random = new Random();
        this.residuals = new List<Sample>(samples);
        this.trees = new List<Tree>();
        this.learningRate = learningRate;
        this.subsampleFraction = subsampleFraction;
        this.maxDepth = maxDepth;
        this.minLeafSize = minLeafSize;
        this.treeType = treeType;
    }

    public void AddTree()
    {
        // take a fractional subsample of residuals
        List<Sample> residualsFraction = residuals.OrderBy(sample => random.NextSingle()).Take((int)((float)residuals.Count * subsampleFraction)).ToList();

        // add a new tree using the fraction
        if (treeType == typeof(OptimalTree))
        {
            trees.Add(new OptimalTree(residualsFraction, maxDepth, minLeafSize));
        }
        else if (treeType == typeof(RandomTree))
        {
            trees.Add(new RandomTree(residualsFraction, maxDepth, minLeafSize));
        }
        else
        {
            throw new Exception("Invalid tree type");
        }

        // update residuals
        Tree lastTree = trees.Last();
        for (int residualIndex = 0; residualIndex < residuals.Count; residualIndex++)
        {
            Sample residual = residuals[residualIndex];
            float[] prediction = lastTree.Predict(residual.input);
            float[] residualOutput = new float[residuals[0].output.Length];
            for (int outputIndex = 0; outputIndex < residuals[0].output.Length; outputIndex++)
            {
                residualOutput[outputIndex] = residual.output[outputIndex] - (prediction[outputIndex] * learningRate);
            }
            residuals[residualIndex] = new Sample(residual.input, residualOutput);
        }
    }

    public float[] Predict(float[] input)
    {
        float[] prediction = new float[residuals[0].output.Length];
        foreach (Tree tree in trees)
        {
            float[] treePrediction = tree.Predict(input);
            for (int outputIndex = 0; outputIndex < prediction.Length; outputIndex++)
            {
                prediction[outputIndex] += treePrediction[outputIndex] * learningRate;
            }
        }
        return prediction;
    }
}

public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            float[] labelOneHot = new float[10];
            labelOneHot[labelInt] = 1;
            float[] input = new float[parts.Length - 1];
            for (int i = 1; i < parts.Length; i++)
            {
                input[i - 1] = float.Parse(parts[i]) / 255f;
            }
            samples.Add(new Sample(input, labelOneHot));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static int Argmax(float[] values)
    {
        int argmax = 0;
        float max = values[0];
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

    public static float ArgmaxError(float[] prediction, List<Sample> samples)
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
        return (float)errorCount / (float)samples.Count;
    }

    public static List<float[]> KNN(List<Sample> train, float[] testInput, int maxK, float exponent)
    {
        // find neighbours
        (Sample trainSample, float distance)[] neighboursArray = new (Sample trainSample, float distance)[train.Count];
        Parallel.For(0, train.Count, trainIndex =>
        {
            float[] trainInput = train[trainIndex].input;
            float distance = 0f;
            for (int i = 0; i < trainInput.Length; i++)
            {
                distance += MathF.Pow(MathF.Abs(trainInput[i] - testInput[i]), exponent);
            }
            distance = MathF.Pow(distance, 1f / exponent);
            neighboursArray[trainIndex] = (train[trainIndex], distance);
        });

        // sort near to far
        List<(Sample trainSample, float distance)> neighbours = neighboursArray.ToList();
        neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));

        // create results
        List<float[]> results = new List<float[]>();

        // iterate through k values
        for (int k = 1; k <= maxK; k++)
        {
            // get the k neighbours
            List<(Sample trainSample, float distance)> kNeighbours = neighbours.Take(k).ToList();

            // find the max distance of the neighbours
            double maxDistance = kNeighbours.Max(neighbour => neighbour.distance) + 0.000001; // epsilon prevents /0 and only 0 weights

            // create the result and weight sum
            float[] result = new float[train[0].output.Length];
            double weightSum = 0f;

            // iterate through neighbours
            foreach ((Sample trainSample, float distance) kNeighbour in kNeighbours)
            {
                // calculate the weight of this neighbour
                double weight = 1 - (kNeighbour.distance / maxDistance);

                // add to weight sum
                weightSum += weight;

                // add to result
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] += kNeighbour.trainSample.output[i] * (float)weight;
                }
            }

            // weighted average result
            for (int i = 0; i < result.Length; i++)
            {
                result[i] /= (float)weightSum;
            }

            // add to results
            results.Add(result);
        }

        // return results
        return results;
    }

    public static void Main(string[] args)
    {
        List<Sample> train = ReadMNIST("d:/data/mnist_train.csv", max: -1);
        List<Sample> test = ReadMNIST("d:/data/mnist_test.csv", max: -1);

        TextWriter log = new StreamWriter("log.csv");
        log.WriteLine("k,exponent,argmax,mae");
        log.Flush();

        int maxK = 100;
        for (float exponent = 1f; exponent < 30f; exponent += 0.02f)
        {
            Console.WriteLine("Exponent: " + exponent);
            int[] kArgmax = new int[maxK];
            float[] kMAE = new float[maxK];
            for (int sampleIndex = 0; sampleIndex < test.Count; sampleIndex++)
            {
                Console.Write($"\rTest: {sampleIndex + 1}/{test.Count}");
                Sample testSample = test[sampleIndex];
                List<float[]> predictions = KNN(train, testSample.input, maxK, exponent);
                float[] actual = testSample.output;
                int actualArgmax = Argmax(actual);
                for (int k = 1; k <= maxK; k++)
                {
                    float[] kPrediction = predictions[k - 1];
                    int kArgmaxPrediction = Argmax(kPrediction);
                    if (kArgmaxPrediction == actualArgmax)
                    {
                        kArgmax[k - 1]++;
                    }
                    for (int i = 0; i < kPrediction.Length; i++)
                    {
                        kMAE[k - 1] += MathF.Abs(kPrediction[i] - actual[i]);
                    }
                }
            }
            Console.WriteLine();
            for (int k = 1; (k <= maxK); k++)
            {
                kMAE[k - 1] /= ((float)test.Count * (float)test[0].output.Length);
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