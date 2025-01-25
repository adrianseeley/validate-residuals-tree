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

    public double distanceBiasMin;
    public double distanceBiasMax;
    public double distanceBiasStep;

    public double distanceExponentMin;
    public double distanceExponentMax;
    public double distanceExponentStep;

    public double distanceRootMin;
    public double distanceRootMax;
    public double distanceRootStep;

    public double weightExponentMin;
    public double weightExponentMax;
    public double weightExponentStep;

    public Aggregation[] aggregations;
    public Neighbour[] neighbours;

    public KNNConfiguration(Dataset dataset, int maxK, double distanceBiasMin, double distanceBiasMax, double distanceBiasStep, double distanceExponentMin, double distanceExponentMax, double distanceExponentStep, double distanceRootMin, double distanceRootMax, double distanceRootStep, double weightExponentMin, double weightExponentMax, double weightExponentStep)
    {
        this.maxK = maxK;

        this.distanceBiasMin = distanceBiasMin;
        this.distanceBiasMax = distanceBiasMax;
        this.distanceBiasStep = distanceBiasStep;

        this.distanceExponentMin = distanceExponentMin;
        this.distanceExponentMax = distanceExponentMax;
        this.distanceExponentStep = distanceExponentStep;

        this.distanceRootMin = distanceRootMin;
        this.distanceRootMax = distanceRootMax;
        this.distanceRootStep = distanceRootStep;

        this.weightExponentMin = weightExponentMin;
        this.weightExponentMax = weightExponentMax;
        this.weightExponentStep = weightExponentStep;

        this.aggregations = [Aggregation.Flat, Aggregation.InverseNormal, Aggregation.Reciprocal];
        this.neighbours = Enumerable.Range(0, dataset.train.Length).Select(trainIndex => new Neighbour()).ToArray();
    }
}