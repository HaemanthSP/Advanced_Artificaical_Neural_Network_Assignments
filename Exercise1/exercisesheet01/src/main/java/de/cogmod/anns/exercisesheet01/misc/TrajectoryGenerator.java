package de.cogmod.anns.exercisesheet01.misc;

public interface TrajectoryGenerator {
    public void reset();
    public void reset(double x, double y);
    public int vectorsize();
    public double[] next();
}