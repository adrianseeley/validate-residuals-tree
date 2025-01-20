public class Sample
{
    public float[] input;
    public float[] output;

    public Sample(float[] input, float[] output)
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
            foreach(float inputValue in inputValues)
            {
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
    public int inputCount;
    public int outputCount;
    public List<Tree> trees;
    public float learningRate;
    public float subsampleFraction;
    public int maxDepth;
    public int minLeafSize;
    public Type treeType;

    public ResidualTrees(int inputCount, int outputCount, float learningRate, float subsampleFraction, int maxDepth, int minLeafSize, Type treeType)
    {
        this.random = new Random();
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.trees = new List<Tree>();
        this.learningRate = learningRate;
        this.subsampleFraction = subsampleFraction;
        this.maxDepth = maxDepth;
        this.minLeafSize = minLeafSize;
        this.treeType = treeType;
    }

    public void AddTree(List<Sample> samples)
    {
        // resample
        List<Sample> resampled = samples.OrderBy(sample => random.NextSingle()).Take((int)((float)samples.Count * subsampleFraction)).ToList();

        // compute up to the current tree worth of residuals
        List<Sample> residuals = new List<Sample>(resampled);
        foreach(Tree tree in trees)
        {
            for (int residualIndex = 0; residualIndex < residuals.Count; residualIndex++)
            {
                Sample residual = residuals[residualIndex];
                float[] prediction = tree.Predict(residual.input);
                float[] residualOutput = new float[outputCount];
                for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
                {
                    residualOutput[outputIndex] = residual.output[outputIndex] - (prediction[outputIndex] * learningRate);
                }
                residuals[residualIndex] = new Sample(residual.input, residualOutput);
            }
        }

        // add a new tree
        if (treeType == typeof(OptimalTree))
        {
            trees.Add(new OptimalTree(residuals, maxDepth, minLeafSize));
        }
        else if (treeType == typeof(RandomTree))
        {
            trees.Add(new RandomTree(residuals, maxDepth, minLeafSize));
        }
        else
        {
            throw new Exception("Invalid tree type");
        }
    }

    public float[] Predict(float[] input)
    {
        float[] prediction = new float[outputCount];
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

    public static void Main(string[] args)
    {
        List<Sample> samples = ReadMNIST("d:/data/mnist_train.csv", max: 2000);
        List<Sample> train = samples.Take(1000).ToList();
        List<Sample> validate = samples.Skip(1000).ToList();

        TextWriter log = new StreamWriter("log.csv");
        log.WriteLine("trees,validateArgmax,validateMae,trainArgmax,trainMae");
        object logLock = new object();

        int maxTrees = 100000;
        float learningRate = 0.01f;
        float subsampleFraction = 0.5f;
        int minLeafSize = 1;
        int maxDepth = int.MaxValue;
       
        ResidualTrees residualTrees = new ResidualTrees(train[0].input.Length, train[0].output.Length, learningRate, subsampleFraction, maxDepth, minLeafSize, typeof(RandomTree));

        for (int treeIndex = 0; treeIndex < maxTrees; treeIndex++)
        {
            residualTrees.AddTree(train);

            int argmaxValidate = 0;
            float maeValidate = 0f;
            foreach (Sample sample in validate)
            {
                float[] prediction = residualTrees.Predict(sample.input);
                int predictionArgmax = Argmax(prediction);
                int sampleArgmax = Argmax(sample.output);
                if (predictionArgmax == sampleArgmax)
                {
                    argmaxValidate++;
                }
                for (int outputIndex = 0; outputIndex < prediction.Length; outputIndex++)
                {
                    maeValidate += MathF.Abs(prediction[outputIndex] - sample.output[outputIndex]);
                }
            }
            maeValidate /= ((float)validate.Count * (float)samples[0].output.Length);

            int argmaxTrain = 0;
            float maeTrain = 0f;
            foreach (Sample sample in train)
            {
                float[] prediction = residualTrees.Predict(sample.input);
                int predictionArgmax = Argmax(prediction);
                int sampleArgmax = Argmax(sample.output);
                if (predictionArgmax == sampleArgmax)
                {
                    argmaxTrain++;
                }
                for (int outputIndex = 0; outputIndex < prediction.Length; outputIndex++)
                {
                    maeTrain += MathF.Abs(prediction[outputIndex] - sample.output[outputIndex]);
                }
            }
            maeTrain /= ((float)train.Count * (float)samples[0].output.Length);

            lock (logLock)
            {
                Console.WriteLine($"TREES: {treeIndex + 1}, VALIDATE: {argmaxValidate}, {maeValidate}, TRAIN: {argmaxTrain}, {maeTrain}");
                log.WriteLine($"{treeIndex + 1},{argmaxValidate},{maeValidate},{argmaxTrain},{maeTrain}");
                log.Flush();
            }
        }
    }
}