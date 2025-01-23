public class Neighbour
{
    public int index;
    public Sample? sample;
    public double distance;

    public Neighbour()
    {
        this.index = -1;
        this.sample = null;
        this.distance = double.MaxValue;
    }
}