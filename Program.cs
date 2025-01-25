public class Program
{
    public static void Main(string[] args)
    {
        int maxK = 10;
        List<Dataset> datasets = new List<Dataset>()
        {
            Data.IRIS("./data/IRIS/iris.data"),
            Data.MNIST("d:/data/mnist_train.csv", "d:/data/mnist_test.csv", 1000),
        };

        foreach (Dataset dataset in datasets)
        {
            KNNConfiguration knnConfiguration = new KNNConfiguration(
                dataset: dataset, 
                maxK: maxK,
                distanceBiasMin: 0.0,
                distanceBiasMax: 7.5,
                distanceBiasStep: 0.25,
                distanceExponentMin: 1.0,
                distanceExponentMax: 30.0,
                distanceExponentStep: 1,
                distanceRootMin: 1.0,
                distanceRootMax: 30.0,
                distanceRootStep: 1,
                weightExponentMin: 1.0,
                weightExponentMax: 30.0,
                weightExponentStep: 1
            );
            KNN.Run(dataset, knnConfiguration);
        }
    }
}