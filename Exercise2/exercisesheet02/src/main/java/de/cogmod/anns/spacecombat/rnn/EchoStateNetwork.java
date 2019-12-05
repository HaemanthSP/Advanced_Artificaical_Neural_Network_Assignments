package de.cogmod.anns.spacecombat.rnn;
import static de.cogmod.anns.spacecombat.rnn.ReservoirTools.*;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    
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
        final int t = this.getLastInputLength() - 1;
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }
    
    /**
     * ESN training algorithm. 
     */
    public double trainESN(
        final double[][] sequence,
        final int washout,
        final int training,
        final int test
    ) {
        //
        // TODO: implement ESN training algorithm here. 
        //
        // Washout 1
        for (int i = 0; i < washout; i++) {
            forwardPassOscillator()
            teacherForcing(sequence[i]);
        }

        // Training
        for (int i = washout; i < training; i++) {
            // Forward pass (feeds the output of previous state as input)
            double[] output = forwardPassOscillator();
            System.out.println();
            System.out.println("Training sequence", i)
            System.out.println("Output");
            System.out.println(matrixAsString(output, 2));
            System.out.println();
            System.out.println("Expected Output");
            System.out.println(matrixAsString(sequence[i], 2));

            // Compute the Wout that could project the reservoir state to output
            // Compute pseudo inverse
            final double[][] reservoirAct = this.getAct()[1];
            solveSvd(reservoirAct, sequence, this.outputweights);
        }

        // Washout 2
        for (int i = 0; i < washout; i++) {
            forwardPassOscillator()
            teacherForcing(sequence[i]);
        }

        // Validation
        for (int i = washout; i < training; i++) {
            // Forward pass (feeds the output of previous state as input)
            double[] output = forwardPassOscillator();
            System.out.println();
            System.out.println("Training sequence", i)
            System.out.println("Output");
            System.out.println(matrixAsString(output, 2));
            System.out.println();
            System.out.println("Expected Output");
            System.out.println(matrixAsString(sequence[i], 2));
        }

        return 0.0; // error.
    }
    
}