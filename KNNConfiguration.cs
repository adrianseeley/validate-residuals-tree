public class KNNConfiguration
{
    public int maxK;
    public double distanceExponent;
    public double distanceRoot;
    public double weightExponent;
    public double[] inputWeights;
    public double[] distanceWeights;
    public Neighbour[] neighbours;

    public KNNConfiguration(Dataset dataset, int maxK)
    {
        this.maxK = maxK;
        this.distanceExponent = 1.0;
        this.distanceRoot = 1.0;
        this.weightExponent = 1.0;
        this.inputWeights = Enumerable.Range(0, dataset.inputLength).Select(input => 1.0).ToArray();
        this.distanceWeights = Enumerable.Range(0, dataset.train.Length).Select(sample => 1.0).ToArray();
        this.neighbours = Enumerable.Range(0, dataset.train.Length).Select(trainIndex => new Neighbour()).ToArray();
    }
}