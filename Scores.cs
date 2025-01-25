public class Scores
{
    public
        Dictionary<int,                           // k
        Dictionary<double,                        // inputDistanceBias
        Dictionary<double,                        // distanceExponent
        Dictionary<double,                        // distanceRoot
        Dictionary<double,                        // weightExponent
        Dictionary<KNNConfiguration.Aggregation,  // aggregation
        Dictionary<bool,                          // isTrain
        int                                       // tally
    >>>>>>> table;

    public Scores()
    {
        table = new Dictionary<int, Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>>>>();
    }

    public void Tally(int k, double inputDistanceBias, double distanceExponent, double distanceRoot, double weightExponent, KNNConfiguration.Aggregation aggregation, bool isTrain)
    {
        if (!table.ContainsKey(k))
        {
            table[k] = new Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>>>();
        }
        Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>>> kTable = table[k];
        if (!kTable.ContainsKey(inputDistanceBias))
        {
            kTable[inputDistanceBias] = new Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>>();
        }
        Dictionary<double, Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>> inputDistanceBiasTable = kTable[inputDistanceBias];
        if (!inputDistanceBiasTable.ContainsKey(distanceExponent))
        {
            inputDistanceBiasTable[distanceExponent] = new Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>>();
        }
        Dictionary<double, Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>> distanceExponentTable = inputDistanceBiasTable[distanceExponent];
        if (!distanceExponentTable.ContainsKey(distanceRoot))
        {
            distanceExponentTable[distanceRoot] = new Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>>();
        }
        Dictionary<double, Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>> distanceRootTable = distanceExponentTable[distanceRoot];
        if (!distanceRootTable.ContainsKey(weightExponent))
        {
            distanceRootTable[weightExponent] = new Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>>();
        }
        Dictionary<KNNConfiguration.Aggregation, Dictionary<bool, int>> weightExponentTable = distanceRootTable[weightExponent];
        if (!weightExponentTable.ContainsKey(aggregation))
        {
            weightExponentTable[aggregation] = new Dictionary<bool, int>();
        }
        Dictionary<bool, int> aggregationTable = weightExponentTable[aggregation];
        if (!aggregationTable.ContainsKey(isTrain))
        {
            aggregationTable[isTrain] = 0;
        }
        aggregationTable[isTrain]++;
    }

    public void Dump(string filename)
    {
        TextWriter log = new StreamWriter(filename);
        log.WriteLine("k,inputDistanceBias,distanceExponent,distanceRoot,weightExponent,aggregation,isTrain,correctTally");
        foreach(int k in table.Keys)
        {
            foreach (double inputDistanceBias in table[k].Keys)
            {
                foreach (double distanceExponent in table[k][inputDistanceBias].Keys)
                {
                    foreach (double distanceRoot in table[k][inputDistanceBias][distanceExponent].Keys)
                    {
                        foreach (double weightExponent in table[k][inputDistanceBias][distanceExponent][distanceRoot].Keys)
                        {
                            foreach (KNNConfiguration.Aggregation aggregation in table[k][inputDistanceBias][distanceExponent][distanceRoot][weightExponent].Keys)
                            {
                                foreach (bool isTrain in table[k][inputDistanceBias][distanceExponent][distanceRoot][weightExponent][aggregation].Keys)
                                {
                                    int tally = table[k][inputDistanceBias][distanceExponent][distanceRoot][weightExponent][aggregation][isTrain];
                                    log.WriteLine($"{k},{inputDistanceBias},{distanceExponent},{distanceRoot},{weightExponent},{aggregation},{isTrain},{tally}");
                                }
                            }
                        }
                    }
                }
            }
        }
        log.Flush();
        log.Close();
    }
}