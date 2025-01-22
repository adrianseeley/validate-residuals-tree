public static class Data
{
    public static List<Sample> IRIS(string filename)
    {
        // iris.data
        List<Sample> samples = new List<Sample>();
        int inputCount = 4;
        int outputCount = 3;
        string[] lines = File.ReadAllLines(filename);
        foreach (string line in lines)
        {
            string[] parts = line.Split(',');
            if (parts.Length != inputCount + 1)
            {
                continue;
            }
            double[] input = new double[inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                input[i] = double.Parse(parts[i]);
            }
            double[] output = new double[outputCount];
            string outputString = parts[inputCount];
            switch (outputString)
            {
                case "Iris-setosa":
                    output[0] = 1;
                    break;
                case "Iris-versicolor":
                    output[1] = 1;
                    break;
                case "Iris-virginica":
                    output[2] = 1;
                    break;
                default:
                    throw new Exception("Invalid class");
            }
            samples.Add(new Sample(input, output));
        }
        return samples;
    }

    public static List<Sample> WINE(string redFilename, string whiteFilename)
    {
        // winequality-red.csv
        // winequality-white.csv
        List<Sample> samples = new List<Sample>();
        int inputCount = 13;
        int outputCount = 1;
        string[] redLines = File.ReadAllLines(redFilename);
        for (int lineIndex = 1; lineIndex < redLines.Length; lineIndex++)
        {
            string line = redLines[lineIndex];
            string[] parts = line.Split(';');
            if (parts.Length != 12)
            {
                continue;
            }
            double[] input = new double[inputCount];
            for (int i = 0; i < 11; i++)
            {
                input[i] = double.Parse(parts[i]);
            }
            // last input is [1, 0] for red, or [0, 1] for  white
            input[11] = 1;
            double[] output = new double[outputCount];
            output[0] = double.Parse(parts[11]);
            samples.Add(new Sample(input, output));
        }
        string[] whiteLines = File.ReadAllLines(whiteFilename);
        for (int lineIndex = 1; lineIndex < whiteLines.Length; lineIndex++)
        {
            string line = whiteLines[lineIndex];
            string[] parts = line.Split(';');
            if (parts.Length != 12)
            {
                continue;
            }
            double[] input = new double[inputCount];
            for (int i = 0; i < 11; i++)
            {
                input[i] = double.Parse(parts[i]);
            }
            // last input is [1, 0] for red, or [0, 1] for  white
            input[11] = 0;
            double[] output = new double[outputCount];
            output[0] = double.Parse(parts[11]);
            samples.Add(new Sample(input, output));
        }
        return samples;
    }

    public static List<Sample> BREAST(string filename)
    {
        //wdbc.data
        List<Sample> samples = new List<Sample>();
        int inputCount = 30;
        int outputCount = 2;
        string[] lines = File.ReadAllLines(filename);
        foreach (string line in lines)
        {
            string[] parts = line.Split(',');
            if (parts.Length != 32)
            {
                continue;
            }
            double[] input = new double[inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                // first col is id, second col is class
                input[i] = double.Parse(parts[i + 2]);
            }
            double[] output = new double[outputCount];
            string outputString = parts[1];
            if (parts[1] == "M")
            {
                output[0] = 1;
            }
            else if (parts[1] == "B")
            {
                output[1] = 1;
            }
            else
            {
                throw new Exception("Invalid class");
            }
            samples.Add(new Sample(input, output));
        }
        return samples;
    }

    public static List<Sample> CAR(string filename)
    {
        // car.data
        List<Sample> samples = new List<Sample>();
        int inputCount = 21;
        int outputCount = 4;
        string[] lines = File.ReadAllLines(filename);
        foreach (string line in lines)
        {
            string[] parts = line.Split(',');
            if (parts.Length != 7)
            {
                continue;
            }
            double[] input = new double[inputCount];
            /*
                buying:   vhigh, high, med, low.
                maint:    vhigh, high, med, low.
                doors:    2, 3, 4, 5more.
                persons:  2, 4, more.
                lug_boot: small, med, big.
                safety:   low, med, high.
                class:    unacc, acc, good, vgood.
            */
            string buying = parts[0];
            string maint = parts[1];
            string doors = parts[2];
            string persons = parts[3];
            string lug_boot = parts[4];
            string safety = parts[5];
            switch (buying)
            {
                case "vhigh":
                    input[0] = 1;
                    break;
                case "high":
                    input[1] = 1;
                    break;
                case "med":
                    input[2] = 1;
                    break;
                case "low":
                    input[3] = 1;
                    break;
                default:
                    throw new Exception("Invalid buying");
            }
            switch (maint)
            {
                case "vhigh":
                    input[4] = 1;
                    break;
                case "high":
                    input[5] = 1;
                    break;
                case "med":
                    input[6] = 1;
                    break;
                case "low":
                    input[7] = 1;
                    break;
                default:
                    throw new Exception("Invalid maint");
            }
            switch (doors)
            {
                case "2":
                    input[8] = 1;
                    break;
                case "3":
                    input[9] = 1;
                    break;
                case "4":
                    input[10] = 1;
                    break;
                case "5more":
                    input[11] = 1;
                    break;
                default:
                    throw new Exception("Invalid doors");
            }
            switch (persons)
            {
                case "2":
                    input[12] = 1;
                    break;
                case "4":
                    input[13] = 1;
                    break;
                case "more":
                    input[14] = 1;
                    break;
                default:
                    throw new Exception("Invalid persons");
            }
            switch (lug_boot)
            {
                case "small":
                    input[15] = 1;
                    break;
                case "med":
                    input[16] = 1;
                    break;
                case "big":
                    input[17] = 1;
                    break;
                default:
                    throw new Exception("Invalid lug_boot");
            }
            switch (safety)
            {
                case "low":
                    input[18] = 1;
                    break;
                case "med":
                    input[19] = 1;
                    break;
                case "high":
                    input[20] = 1;
                    break;
                default:
                    throw new Exception("Invalid safety");
            }
            double[] output = new double[outputCount];
            string outputString = parts[6];
            switch (outputString)
            {
                case "unacc":
                    output[0] = 1;
                    break;
                case "acc":
                    output[1] = 1;
                    break;
                case "good":
                    output[2] = 1;
                    break;
                case "vgood":
                    output[3] = 1;
                    break;
                default:
                    throw new Exception("Invalid class");
            }
            samples.Add(new Sample(input, output));
        }
        return samples;
    }

    public static List<Sample> ABALONE(string filename)
    {
        // abalone.data
        List<Sample> samples = new List<Sample>();
        int inputCount = 10;
        int outputCount = 1;
        string[] lines = File.ReadAllLines(filename);
        foreach (string line in lines)
        {
            string[] parts = line.Split(',');
            if (parts.Length != 9)
            {
                continue;
            }
            double[] input = new double[inputCount];
            string sex = parts[0];
            switch (sex)
            {
                case "M":
                    input[0] = 1;
                    break;
                case "F":
                    input[1] = 1;
                    break;
                case "I":
                    input[2] = 1;
                    break;
                default:
                    throw new Exception("Invalid sex");
            }
            for (int i = 1; i < 8; i++)
            {
                input[i + 2] = double.Parse(parts[i]);
            }
            double[] output = new double[outputCount];
            output[0] = double.Parse(parts[8]);
            samples.Add(new Sample(input, output));
        }
        return samples;
    }

    public static List<Sample> BEAN(string filename)
    {
        // Dry_Bean_Dataset.arff
        List<Sample> samples = new List<Sample>();
        int inputCount = 16;
        int outputCount = 7;
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 25; lineIndex < lines.Length; lineIndex++)
        {
            string[] parts = lines[lineIndex].Split(',');
            if (parts.Length != 17)
            {
                continue;
            }
            double[] input = new double[inputCount];
            for (int i = 0; i < 16; i++)
            {
                input[i] = double.Parse(parts[i]);
            }
            double[] output = new double[outputCount];
            string outputString = parts[16];
            switch (outputString)
            {
                case "SEKER":
                    output[0] = 1;
                    break;
                case "BARBUNYA":
                    output[1] = 1;
                    break;
                case "BOMBAY":
                    output[2] = 1;
                    break;
                case "CALI":
                    output[3] = 1;
                    break;
                case "HOROZ":
                    output[4] = 1;
                    break;
                case "SIRA":
                    output[5] = 1;
                    break;
                case "DERMASON":
                    output[6] = 1;
                    break;
                default:
                    throw new Exception("Invalid class");
            }
            samples.Add(new Sample(input, output));
        }
        return samples;
    }
}