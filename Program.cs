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

public class Tree
{
    public float[] prediction;
    public float error;
    public int? inputIndex;
    public float? inputValue;
    public Tree? left;
    public Tree? right;

    private static float[] DeterminePrediction(List<Sample> samples)
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

    private static float DetermineError(float[] prediction, List<Sample> samples)
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

    public Tree(List<Sample> samples, int maxDepth, int minLeafSize, int currentDepth = 1, bool verbose = false)
    {
        int inputCount = samples[0].input.Length;
        prediction = DeterminePrediction(samples);
        error = DetermineError(prediction, samples);
        if (samples.Count <= minLeafSize * 2 || currentDepth >= maxDepth)
        {
            inputIndex = null;
            inputValue = null;
            left = null;
            right = null;
            return;
        }
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
                if (verbose)
                {
                    Console.Write($"\rSamples: {samples.Count} Index: {inputIndex + 1}/{inputCount}");
                }
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
        if (verbose)
        {
            Console.WriteLine();
        }
        if (bestInputIndex == -1)
        {
            inputIndex = null;
            inputValue = null;
            left = null;
            right = null;
            return;
        }
        inputIndex = bestInputIndex;
        inputValue = bestInputValue;
        leftSamples.Clear();
        rightSamples.Clear();
        foreach (Sample sample in samples)
        {
            if (sample.input[inputIndex.Value] < inputValue.Value)
            {
                leftSamples.Add(sample);
            }
            else
            {
                rightSamples.Add(sample);
            }
        }
        left = new Tree(leftSamples, maxDepth, minLeafSize, currentDepth + 1, verbose);
        right = new Tree(rightSamples, maxDepth, minLeafSize, currentDepth + 1, verbose);
    }

    public float[] Predict(float[] input)
    {
        if (inputIndex == null || inputValue == null)
        {
            return prediction;
        }
        if (left == null || right == null)
        {
            throw new Exception("Invalid tree");
        }
        if (input[inputIndex.Value] < inputValue.Value)
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
        if (inputIndex == null || inputValue == null)
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
            if (validationSample.input[inputIndex.Value] < inputValue.Value)
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
            inputIndex = null;
            inputValue = null;
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
            inputIndex = null;
            inputValue = null;
            left = null;
            right = null;
        }
        return validationError;
    }
}

public class ResidualTrees
{
    public float[] initialPrediction;
    public List<Tree> trees;
    public List<Sample> residuals;
    public float learningRate;
    public int maxDepth;
    public int minLeafSize;

    public ResidualTrees(List<Sample> samples, float learningRate, int maxDepth, int minLeafSize, bool verbose = false)
    {
        initialPrediction = new float[samples[0].output.Length];
        foreach (Sample sample in samples)
        {
            for (int outputIndex = 0; outputIndex < initialPrediction.Length; outputIndex++)
            {
                initialPrediction[outputIndex] += sample.output[outputIndex];
            }
        }
        for (int outputIndex = 0; outputIndex < initialPrediction.Length; outputIndex++)
        {
            initialPrediction[outputIndex] /= (float)samples.Count;
        }
        trees = new List<Tree>();
        residuals = new List<Sample>(samples.Count);
        foreach (Sample sample in samples)
        {
            float[] residualOutput = new float[sample.output.Length];
            for (int outputIndex = 0; outputIndex < residualOutput.Length; outputIndex++)
            {
                residualOutput[outputIndex] = sample.output[outputIndex] - initialPrediction[outputIndex];
            }
            residuals.Add(new Sample(sample.input, residualOutput));
        }
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.minLeafSize = minLeafSize;
    }

    public void AddTree(bool verbose = false)
    {
        Tree tree = new Tree(residuals, maxDepth, minLeafSize, verbose: verbose);
        trees.Add(tree);
        List<Sample> newResiduals = new List<Sample>(residuals.Count);
        foreach (Sample residual in residuals)
        {
            float[] prediction = tree.Predict(residual.input);
            float[] residualOutput = new float[residual.output.Length];
            for (int outputIndex = 0; outputIndex < residualOutput.Length; outputIndex++)
            {
                residualOutput[outputIndex] = residual.output[outputIndex] - (prediction[outputIndex] * learningRate);
            }
            newResiduals.Add(new Sample(residual.input, residualOutput));
        }
        residuals = newResiduals;
    }

    public float[] Predict(float[] input)
    {
        float[] prediction = new float[residuals[0].output.Length];
        Array.Copy(initialPrediction, prediction, prediction.Length);
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
        log.WriteLine("minLeafSize,trees,argmax,mae");
        object logLock = new object();

        int maxTrees = 100;
        float learningRate = 0.05f;

        List<int> minLeafSizes = new List<int>();
        for (int minLeafSize = 1; minLeafSize < 10; minLeafSize++)
        {
            minLeafSizes.Add(minLeafSize);
        }

        Parallel.ForEach(minLeafSizes, minLeafSize =>
        {
            ResidualTrees residualTrees = new ResidualTrees(train, learningRate, int.MaxValue, minLeafSize);
            for (int treeIndex = 0; treeIndex < maxTrees; treeIndex++)
            {
                residualTrees.AddTree();

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

                lock (logLock)
                {
                    Console.WriteLine($"MINLEAFSIZE: {minLeafSize}, TREES: {treeIndex + 1}, ARGMAX: {residualTrees}, MAE: {maeValidate}");
                    log.WriteLine($"{minLeafSize},{treeIndex + 1},{argmaxValidate},{maeValidate}");
                    log.Flush();
                }
            }
        });
    }
}