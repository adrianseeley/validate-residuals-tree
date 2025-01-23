public class Dataset
{
    public string name;
    public Sample[] train;
    public Sample[] test;
    public int[] trainArgmax;
    public int[] testArgmax;
    public int inputLength;
    public int outputLength;

    public Dataset(string name, Sample[] train, Sample[] test)
    {
        this.name = name;
        this.train = train;
        this.test = test;
        this.trainArgmax = train.Select(sample => Utility.Argmax(sample.output)).ToArray();
        this.testArgmax = test.Select(sample => Utility.Argmax(sample.output)).ToArray();
        this.inputLength = train[0].input.Length;
        this.outputLength = train[0].output.Length;
    }
}