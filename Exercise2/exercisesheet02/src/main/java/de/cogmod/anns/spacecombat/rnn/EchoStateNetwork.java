package de.cogmod.anns.spacecombat.rnn;

import java.util.Arrays;

import static de.cogmod.anns.spacecombat.rnn.ReservoirTools.*;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    public int reservoirsize;

    public static double sq(final double x) {
        return x * x;
    }

    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }
    
    public double[][] getOutputWeights() {
        return this.outputweights;
    }
    
    public EchoStateNetwork(
        final int input,
        final int reservoirsize,
        final int output
    ) {
        super(input, reservoirsize, output);
        this.reservoirsize = reservoirsize;
        //
        this.inputweights     = this.getWeights()[0][1];
        this.reservoirweights = this.getWeights()[1][1];
        this.outputweights    = this.getWeights()[1][2];
        //
    }
    
    @Override
    public void rebufferOnDemand(int sequencelength) {
        super.rebufferOnDemand(1);
    }
    
    /**
     * Returns the 
     */
    public double[] output() {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            result[i] = act[outputlayer][i][t];
        }
        //
        return result;
    }
    
    /**
     * This is an ESN specific forward pass realizing 
     * an oscillator by means of an output feedback via
     * the input layer. This method requires that the input
     * layer size matches the output layer size. 
     */
    public double[] forwardPassOscillator() {
        //
        // this method causes an additional copy operation
        // but it is more readable from outside.
        //
        final double[] output = this.output();
        return this.forwardPass(output);
    }
    
    /**
     * Overwrites the current output with the given target.
     */
    public void teacherForcing(final double[] target) {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }

    /**
     * Train output weights of ESN. Used as a fitness measure
     * inside the outer loop of Differential Evolution.
     */
    public double trainESN(
        final double[][] washout,
        final double[][] train
    ) {
        this.reset();
        for (int t = 0; t < washout.length; t++) {
            forwardPassOscillator();
            teacherForcing(washout[t]);
        }

        final double[][] reservoirAct = new double[train.length][this.reservoirsize + 1];

        for (int t = 0; t < train.length; t++) {
            forwardPassOscillator();
            for (int i = 0; i < this.reservoirsize; ++i) {
                reservoirAct[t][i] = this.getAct()[1][i][0];
            }
        }
        //System.out.print("cols(this.outputweights) = ");
        //System.out.println(cols(this.outputweights));
        //System.out.print("rows(this.outputweights) = ");
        //System.out.println(rows(this.outputweights));
        final double[][] copyTrain = new double[train.length][train[0].length];
        for (int t = 0; t < train.length; ++t) {
            for (int i = 0; i < train[0].length; ++i) {
                copyTrain[t][i] = train[t][i];
            }
        }

        solveSVD(reservoirAct, copyTrain, this.outputweights);
        this.reset();

        for (int t = 0; t < washout.length; t++) {
            forwardPassOscillator();
            teacherForcing(washout[t]);
        }

        double mse = 0.0;

        for (int t = 0; t < train.length; ++t) {
            double[] output = forwardPassOscillator();
            for (int i = 0; i < output.length; ++i) {
                mse += sq(train[t][i] - output[i]);
            }
        }

        mse /= (double)train.length;

        return Math.sqrt(mse);
    }

    public double evaluateESN(final double[][] washout, final double[][] eval, String name) {
        this.reset();
        for (int t = 0; t < washout.length; ++t) {
            forwardPassOscillator();
            teacherForcing(washout[t]);
        }

        double mse = 0.0;

        System.out.println("import matplotlib.pyplot as plt");
        System.out.println("output = [");
        for (int t = 0; t < eval.length; ++t) {
            double[] output = forwardPassOscillator();
            System.out.printf("%s,\n", Arrays.toString(output));
            for (int i = 0; i < output.length; ++i) {
                mse += sq(eval[t][i] - output[i]);
            }
        }
        System.out.println("]");
        System.out.println("eval = [");
        for (int t = 0; t < eval.length; ++t) {
            System.out.printf("%s,\n", Arrays.toString(eval[t]));
        }
        System.out.println("]");
        System.out.println("plt.plot([i[0] for i in output]); plt.plot([i[0] for i in eval]);");
        System.out.printf("plt.savefig('%s_x.png'); plt.clf();\n", name);
        System.out.println("plt.plot([i[1] for i in output]); plt.plot([i[1] for i in eval]);");
        System.out.printf("plt.savefig('%s_y.png'); plt.clf();\n", name);
        System.out.println("plt.plot([i[2] for i in output]); plt.plot([i[2] for i in eval]);");
        System.out.printf("plt.savefig('%s_z.png'); plt.clf();\n", name);

        mse /= (double)eval.length;

        return Math.sqrt(mse);
    }
    
    public void printWeights() {
        System.out.printf("double[][] inputweights = %s;\n", Arrays.deepToString(this.inputweights));
        System.out.printf("double[][] outputweights = %s;\n", Arrays.deepToString(this.outputweights));
        System.out.printf("double[][] reservoirweights = %s;\n", Arrays.deepToString(this.reservoirweights));
    }
}
