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
    public delegate float ErrorFunction(float[] prediction, List<Sample> samples);
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

    public Tree(List<Sample> samples, ErrorFunction errorFunction)
    {
        int inputCount = samples[0].input.Length;
        prediction = DeterminePrediction(samples);
        error = errorFunction(prediction, samples);
        if (samples.Count <= 2)
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
                if (leftSamples.Count == 0 || rightSamples.Count == 0)
                {
                    continue;
                }
                float leftError = errorFunction(DeterminePrediction(leftSamples), leftSamples);
                float rightError = errorFunction(DeterminePrediction(rightSamples), rightSamples);
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
        left = new Tree(leftSamples, errorFunction);
        right = new Tree(rightSamples, errorFunction);
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
        Tree tree = new Tree(train, ArgmaxError);

        int trainCorrect = 0;
        int validateCorrect = 0;
        foreach (Sample sample in train)
        {
            int predictionArgmax = Argmax(tree.Predict(sample.input));
            int sampleArgmax = Argmax(sample.output);
            if (predictionArgmax == sampleArgmax)
            {
                trainCorrect++;
            }
        }
        foreach (Sample sample in validate)
        {
            int predictionArgmax = Argmax(tree.Predict(sample.input));
            int sampleArgmax = Argmax(sample.output);
            if (predictionArgmax == sampleArgmax)
            {
                validateCorrect++;
            }
        }
        Console.WriteLine("Train accuracy: " + ((float)trainCorrect / (float)train.Count) + " (" + trainCorrect + " / " + train.Count + ")");
        Console.WriteLine("Validate accuracy: " + ((float)validateCorrect / (float)validate.Count) + " (" + validateCorrect + " / " + validate.Count + ")");
    }
}