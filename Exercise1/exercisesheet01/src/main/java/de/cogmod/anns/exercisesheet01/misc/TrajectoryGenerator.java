package de.cogmod.anns.exercisesheet01.misc;

public interface TrajectoryGenerator {
    public void reset();
    public int vectorsize();
    public double[] next();
}