package de.cogmod.anns.exercisesheet01;

import java.util.Arrays;
import java.util.Random;

import de.cogmod.anns.exercisesheet01.misc.LearningListener;
import de.cogmod.anns.exercisesheet01.misc.Tools;

/**
 * Simple implementation of a recurrent neural network with 
 * hyperbolic tangent neurons and trainable biases.
 * 
 * @author Sebastian Otte
 */
public class RecurrentNeuralNetwork {
    
    public final static double BIAS = 1.0;
    
    private int layersnum;
    private int inputsize;
    private int weightsnum;
    private int[] layer;
    private double[][][] net;
    private double[][][] act;
    private double[][][] bwbuffer;
    private double[][][] delta;
    private double[][][][] weights;
    private boolean[] usebias;
    private double[][][][] dweights;
    
    private int bufferlength    = 0;
    private int lastinputlength = 0;
    
    private static int[] join(final int i1, final int i2, final int ...in) {
        final int[] result = new int[2 + in.length];
        //
        result[0] = i1;
        result[1] = i2;
        //
        for (int i = 0; i < in.length; i++) {
            result[i + 2] = in[i];
        }
        //
        return result;
    }
    
    public void setBias(final int layer, final boolean bias) {
        assert(layer > 0 && layer < this.layersnum);
        this.usebias[layer] = bias;
    }
    
    public boolean getBias(final int layer) {
        assert(layer > 0 && layer < this.layersnum);
        return this.usebias[layer];
    }
    
    /**
     * Constructor of the RNN class. The function signature ensures that there is
     * at least an input and an output layer (in this case layern.length would be 0). 
     * Otherwise layer2 would be the first hidden layer and layern would contain
     * at least the output layer and, optionally, further hidden layers. Effectively,
     * the firstly given layer defines the input layer size, the lastly given number
     * defines the output layer size, and the numbers in between define the hidden 
     * layers sizes, accordingly.
     * 
     * @param input Number of input neurons.
     * @param layer2 Layer of hidden/output neurons.
     * @param layern A list of further hidden and the output layer.
     */
    public RecurrentNeuralNetwork(
        final int input, final int layer2, final int... layern 
    ) {
        //
        this.inputsize = input;
        this.layer     = join(input, layer2, layern);
        this.layersnum = this.layer.length;
        //
        // set up buffers.
        //
        this.act      = new double[this.layersnum][][];
        this.net      = new double[this.layersnum][][];
        this.delta    = new double[this.layersnum][][];
        this.bwbuffer = new double[this.layersnum][][];
        //
        this.rebufferOnDemand(1);
        //
        this.usebias  = new boolean[this.layersnum];
        this.weights  = new double[this.layersnum][this.layersnum][][];
        this.dweights = new double[this.layersnum][this.layersnum][][];
        //
        int sumweights = 0;
        //
        for (int l = 0; l < this.layersnum; l++) {
            //
            this.usebias[l] = true;
            //
            // The weights buffer works diffently compared to the MLP:
            //   o its 4-dimension
            //   o the first 2 indices address the source and the destination
            //     layer matrix, respectively.
            //   o the last 2 indices address the connections in the usual manner
            // 
            // For instance, this.weights[0][1][4][6] this address connection from
            // neuron 4 of layer 0 to neuron 6 of layer 1. 
            //
            // Note that in this implementation only the forward weight matrices [l][l+1]
            // and and recurrent weight matrices [l][l] (for non-input non-output layers)
            // are defined -> all others are null.
            //
            if (l > 0) {
                //
                // forward weights. the plus + 1 used to model the bias weights.
                //
                this.weights[l-1][l]  = new double[this.layer[l - 1] + 1][this.layer[l]];
                this.dweights[l-1][l] = new double[this.layer[l - 1] + 1][this.layer[l]];
                sumweights += (this.layer[l - 1] + 1) * (this.layer[l]);
                //
                // if the current layer is a hidden layer, also add recurrent connections.
                //
                if (l < (this.layersnum - 1)) {
                    this.weights[l][l]  = new double[this.layer[l]][this.layer[l]];
                    this.dweights[l][l] = new double[this.layer[l]][this.layer[l]];
                    //
                    sumweights += (this.layer[l] * this.layer[l]);
                }
            }
            //
        }
        //
        this.weightsnum = sumweights;
    }
    
    private static double tanh(final double x) {
        return Math.tanh(x);
    }
    
    private static double tanhDx(final double x) {
        final double tanhx = Math.tanh(x);
        return 1 - (tanhx * tanhx);
        
    }

    public void reset() {
        for (int l = 0; l < this.layersnum; l++) {
            //
            for (int i = 0; i < this.layer[l]; i++) {
                for (int t = 0; t < this.bufferlength; t++) {
                    this.act[l][i][t]      = 0.0;
                    if (l > 0) {
                        this.net[l][i][t]      = 0.0;
                        this.delta[l][i][t]    = 0.0;
                        this.bwbuffer[l][i][t] = 0.0;
                    }
                }
            }
        }
        //
        this.lastinputlength = 0;
    }

    
    
    public void rebufferOnDemand(final int sequencelength) {
        //
        if (this.bufferlength != sequencelength) {
            for (int l = 0; l < this.layersnum; l++) {
                //
                if (l > 0) {
                    //
                    // we don't need the net buffer for input neurons.
                    //
                    this.net[l]      = new double[this.layer[l]][sequencelength];
                    this.delta[l]    = new double[this.layer[l]][sequencelength];
                    this.bwbuffer[l] = new double[this.layer[l]][sequencelength];
                }
                this.act[l] = new double[this.layer[l]][sequencelength];
            }
        }
        //
        this.bufferlength    = sequencelength;
        this.lastinputlength = 0;
    }
    

    /**
     * Computes the forward pass, i.e., propagates an input 
     * vector through the network to the output layer. This is
     * a wrapper method for one time step online.
     * @param input Input vector.
     * @return Output vector.
     */
    public double[] forwardPass(final double[] input) {
        return forwardPass(new double[][]{input})[0];
    }
    
    /**
     * Computes the forward pass, i.e., propagates a sequence
     * of input vectors through the network to the output layer.
     * @param input Sequence of input vectors.
     * @return Output Sequence of output vectors.
     */
    public double[][] forwardPass(final double[][] input) {
        //
        final int sequencelength = Math.min(this.bufferlength, input.length);
        final double[][] output  = new double[sequencelength][];
        final int outputlayer    = this.layersnum - 1;
        //
        int prevt = 0;
        //
        for (int t = 0; t < sequencelength; t++) {
            //
            // store input.
            //
            //System.out.println(Arrays.toString(input[0]));
            assert(input[t].length == this.inputsize);
            for (int i = 0; i < input[t].length; i++) {
                this.act[0][i][t] = input[t][i];
            }
            //
            // compute output layer-wise. start with the first
            // hidden layer (or the outputlayer if there is no
            // hidden layer).
            //
            for (int l = 1; l < this.layersnum; l++) {
                //
                // first compute all the net (integrate the inputs) values and activations.
                //
                final int layersize    = this.layer[l];
                final int prelayersize = this.layer[l - 1];
                //
                //
                final double[][] ff_weights = this.weights[l - 1][l];
                final double[][] fb_weights = this.weights[l][l];
                //
                for (int j = 0; j < layersize; j++) {
                    //
                    // eventually initialize netjt with the weighted bias.
                    //
                    double netjt = 0;
                    //
                    if (this.usebias[l]) {
                        netjt = BIAS * ff_weights[prelayersize][j];
                    }
                    //
                    // integrate feed-forward input.
                    //
                    for (int i = 0; i < prelayersize; i++) {
                        netjt += this.act[l - 1][i][t] * ff_weights[i][j];
                    }
                    //
                    if (l < outputlayer) {
                        //
                        // integrate recurrent input.
                        //
                        // Fixme: Doubt: when t == 0 | activation at layer l is empty | Its fine because of the zero initialization
                        for (int i = 0; i < layersize; i++) {
                            netjt += this.act[l][i][prevt] * fb_weights[i][j];
                        }
                    }
                    //  
                    this.net[l][j][t] = netjt;
                }
                //
                // now we compute the activations of the neurons in the current layer.
                //
                if (l < outputlayer) {
                    //
                    // tanh hidden layer.
                    //
                    for (int j = 0; j < layersize; j++) {
                        this.act[l][j][t] = tanh(this.net[l][j][t]);
                    }    
                } else {
                    //
                    // linear output layer.
                    //
                    for (int j = 0; j < layersize; j++) {
                        this.act[l][j][t] = this.net[l][j][t];
                    }    
                }
            }
            //
            // store output.
            //
            final int outputlayersize = this.layer[outputlayer];
            //
            output[t] = new double[outputlayersize];
            //
            for (int k = 0; k < outputlayersize; k++) {
                output[t][k] = this.act[outputlayer][k][t];
            }
            if (t > 0) prevt = t;
        }
        //
        // Store input length of the current sequence. You can 
        // use this information in the backward pass, i.e., starting the
        // back propagation through time procedure at this index,
        // which can be smaller than the current buffer length
        // depending on the current input sequence.
        //
        this.lastinputlength = sequencelength;
        //
        // return output layer activation as output.
        //
        return output;
    }
    
    
    public void backwardPass(final double[][] target) {
        //
        final int outputlayer = this.delta.length - 1;
        final int steps       = this.lastinputlength;
        //
        int t_target = target.length - 1;
        //
        // compute reversely in time.
        //
        // System.out.println("target: " + target.length);
        // System.out.println("steps: " + steps);
        for (int t = (steps - 1); t >= 0; t--) {
            //
            // inject the output/target discrepancy into this.bwbuffer. Note that
            // this.bwbuffer functions analogously to this.net but is used to
            // store into "back-flowing" inputs (deltas).
            //

            // FIXME: Doubt | difference between t_target and t ? | So the steps and target could be of different size?
            if (t_target >= 0) {
                for (int j = 0; j < this.delta[outputlayer].length; j++) {
                    this.bwbuffer[outputlayer][j][t] = (target[t_target][j] - this.act[outputlayer][j][t]);

                    // delata is same as bwbuffer for output. Because of the linear activation for the output layer.
                    this.delta[outputlayer][j][t] = this.bwbuffer[outputlayer][j][t];
                }
            }
            //
            // back-propagate the error through the network -- we compute the deltas --
            // starting with the output layer.
            //

            for (int l = outputlayer; l > 1; l--) {

                final int layersize    = this.layer[l];
                final int prelayersize = this.layer[l - 1];

                //
                // integrate deltas for non-output layers.
                //
                for (int h = 0; h < prelayersize; h++) {
                    for (int k = 0; k < layersize; k++) {
                        this.bwbuffer[l-1][h][t] += this.weights[l-1][l][h][k] * this.delta[l][k][t];
                    }

                    if (t < (steps - 1)) {
                        for (int m = 0; m < prelayersize; m++) {
                            this.bwbuffer[l - 1][h][t] += this.weights[l - 1][l - 1][h][m] * this.delta[l - 1][m][t + 1];
                        }
                    }

                    this.delta[l - 1][h][t] = this.bwbuffer[l - 1][h][t] * tanhDx(this.net[l - 1][h][t]);
                }
            }

            t_target--;
        }
        // 
        // Compute the weights derivatives.
        //

        for (int l = 1; l <= outputlayer; l++) {

            final int layersize    = this.layer[l];
            final int prelayersize = this.layer[l - 1];

            for (int j = 0; j < layersize; j++) {
                for (int t = 0; t < steps; t++) {
                    for (int i = 0; i < prelayersize; i++) {
                        //
                        // compute weights derivatives between previous layer and current layer.
                        //
                        this.dweights[l-1][l][i][j] += (this.act[l-1][i][t] * this.delta[l][j][t]);
                    }

                    for (int h = 0; h < layersize; h++) {
                        //
                        // compute weights derivatives between current layer and current layer.
                        //
                        if (l < outputlayer && t > 0) {
                            // System.out.println("layer:" + l);
                            // System.out.println("t:" + t);
                            // System.out.println("j:" + j);
                            // System.out.println("h:" + h);
                            // System.out.println("delta:" + this.delta[l][j][t]);
                            // System.out.println("act:" + this.act[l][h][t - 1]);

                            this.dweights[l][l][h][j] += (this.act[l][h][t - 1] * this.delta[l][j][t]);
                        }
                    }

                    //
                    // compute weights derivatives between bias and current layer.
                    //
                    this.dweights[l-1][l][prelayersize][j] +=  this.delta[l][j][t];
                }
            }
        }
    }

    /**
     * Initializes the weights randomly and normal distributed with
     * std. dev. 0.1.
     * @param rnd Instance of Random.
     */
    public void initializeWeights(final Random rnd, final double stddev) {
        for (int l1 = 0; l1 < this.weights.length; l1++) {
            for (int l2 = 0; l2 < this.weights[l1].length; l2++) {
                double[][] wll = this.weights[l1][l2];
                if (wll != null) {
                    for (int i = 0; i < wll.length; i++) {
                        for (int j = 0; j < wll[i].length; j++) {
                            wll[i][j] = rnd.nextGaussian() * stddev;
                        }
                    }
                }
            }
        }
    }
    
    public int getWeightsNum() {
        return this.weightsnum;
    }
    
    
    
    private static void map(final double[] from, final double[][][][] to) {
        int idx = 0;
        for (int l1 = 0; l1 < to.length; l1++) {
            for (int l2 = 0; l2 < to[l1].length; l2++) {
                double[][] wll = to[l1][l2];
                if (wll != null) {
                    for (int i = 0; i < wll.length; i++) {
                        for (int j = 0; j < wll[i].length; j++) {
                            wll[i][j] = from[idx++];
                        }
                    }
                }
            }
        }
    }
    
    private static void map(final double[][][][] from, final double[] to) {
        int idx = 0;
        for (int l1 = 0; l1 < from.length; l1++) {
            for (int l2 = 0; l2 < from[l1].length; l2++) {
                double[][] wll = from[l1][l2];
                if (wll != null) {
                    for (int i = 0; i < wll.length; i++) {
                        for (int j = 0; j < wll[i].length; j++) {
                            to[idx++] = wll[i][j];
                        }
                    }
                }
            }
        }
    }

    public void writeWeights(final double[] weights) {
        map(weights, this.weights);
    }
    
    public void readWeights(final double[] weights) {
        map(this.weights, weights);
    }
                
    public void readDWeights(final double[] dweights) {
        map(this.dweights, dweights);
    }
    
    /**
     * Stochastic gradient descent.
     * @param rnd Instance of Random.
     * @param input Input vectors.
     * @param target Target vectors.
     * @param epochs Number of epochs.
     * @param learningrate Value for the learning rate.
     * @param momentumrate Value for the momentum rate.
     * @param listener Listener to observe the training progress.
     * @return The final epoch error.
     */    
    public double trainStochastic(
        final Random rnd, 
        final double[][][] input, 
        final double[][][] target,
        final double epochs,
        final double learningrate,
        final double momentumrate,
        final LearningListener listener
    ) {
        //
        assert(input.length == target.length);
        //
        final double[] weights       = new double[this.weightsnum];
        final double[] dweights      = new double[this.weightsnum];
        final double[] weightsupdate = new double[this.weightsnum];
        //
        this.readWeights(weights);
        //
        // create initial index permutation.
        //
        final int[] indices = new int[input.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        //
        double error = Double.POSITIVE_INFINITY;
        //
        // epoch loop.
        //
        for (int i = 0; i < epochs; i++) {

            // Reset the deltas and activities from previous epoch
            this.reset();
            //
            // shuffle indices.
            //
            Tools.shuffle(indices, rnd);
            //
            double errorsum = 0.0;

            for (int j = 0; j < input.length; j++) {

                // Forwardpass
                double[][] output = forwardPass(input[indices[j]]);

                // Calculate error
                errorsum += RMSE(output, target[indices[j]]);

                // Backpropagate errors through the network
                backwardPass(target[indices[j]]);
                this.readDWeights(dweights);

                // Update weights
                for (int u = 0; u < this.weightsnum; u++) {
                    weightsupdate[u] = -learningrate * dweights[u] + momentumrate * weightsupdate[u];
                    weights[u] += weightsupdate[u];
                }
                writeWeights(weights);
            }
            //
            // train all samples in online manner, i.e. iterate over all samples
            // while considering the shuffled order and update the weights 
            // immediately after each sample
            //

            // ...
        
            error = errorsum / (double)(input.length);
            if (listener != null) listener.afterEpoch(i + 1, error);
        }
        //
        return error;
    }
    
    /**
     * Computes the RMSE of the current output and
     * a given target vector.
     * @param target Target vector.
     * @return RMSE value.
     */
    public static double RMSE(final double[][] output, final double[][] target) {
        //
        final int length = Math.min(output.length, target.length);
        //
        double error = 0;
        int    ctr   = 0;
        //
        for (int t = 0; t < length; t++) {
            assert(output[t].length > 0);
            assert(target[t].length > 0);
            assert(target[t].length == output[t].length);
            //
            for (int i = 0; i < target[t].length; i++) {
                final double e = output[t][i] - target[t][i];
                error += (e * e);
                ctr++;
            }
        }
        //
        return Math.sqrt(error / (double)(ctr));
    }
    
    
}