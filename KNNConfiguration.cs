public class KNNConfiguration
{
    public const double EPSILON = 0.0000001;

    public enum Aggregation
    {
        Flat,
        InverseNormal,
        Reciprocal
    }

    public int maxK;
    public double inputDistanceBias;
    public double distanceExponent;
    public double distanceRoot;
    public double weightExponent;
    public Aggregation aggregation;
    public Neighbour[] neighbours;

    public KNNConfiguration(Dataset dataset, int maxK)
    {
        this.maxK = maxK;
        this.inputDistanceBias = 0.0;
        this.distanceExponent = 1.0;
        this.distanceRoot = 1.0;
        this.weightExponent = 1.0;
        this.aggregation = Aggregation.Flat;
        this.neighbours = Enumerable.Range(0, dataset.train.Length).Select(trainIndex => new Neighbour()).ToArray();
    }
}