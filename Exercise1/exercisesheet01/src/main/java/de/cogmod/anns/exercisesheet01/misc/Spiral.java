package de.cogmod.anns.exercisesheet01.misc;



public class Spiral implements TrajectoryGenerator {

    private double scale; 
    private double t;
    
    public Spiral() {
        this.reset();
    }
    
    @Override
    public int vectorsize() {
        return 2;
    }
    
    @Override
    public void reset() {
        this.scale = 1.0;
        this.t     = 0.0;
    }

    @Override
    public void reset(double x, double y) {
      this.scale = Math.sqrt(x*x + y*y);
      this.t = Math.atan2(y, x);
    }

    @Override
    public double[] next() {
        final double[] result = new double[2];
        result[0] = Math.cos(this.t) * this.scale;
        result[1] = Math.sin(this.t) * this.scale;
        this.scale *= 0.99;
        this.t += 0.1;
        return result;
    }
    
}