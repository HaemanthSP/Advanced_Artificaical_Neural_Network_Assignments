package de.cogmod.anns.exercisesheet01;

import java.util.Random;

import de.cogmod.anns.exercisesheet01.MultiLayerPerceptron;
import de.cogmod.anns.exercisesheet01.misc.BasicLearningListener;

/**
 * @author Sebastian Otte
 */
public class MLPXOR {
    
    public static void main(String[] args) {
        //
        final double[][] input = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        final double[][] target = {
                {1}, {1}, {1}, {1}
        };
        //
        final Random rnd = new Random(100L);
        //
        // set up network. biases are used by default, but
        // be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = the first hidden layer).
        //
        final MultiLayerPerceptron net = new MultiLayerPerceptron(2, 2,  1);

        //
        // perform training.
        //
        final int epochs = 10000;         // don't change this value!
        final double learningrate = 0.01;
        final double momentumrate = 0.95;
        //
        // generate initial weights.
        //
        net.initializeWeights(rnd, 0.1);
        //
        net.trainStochastic(
            rnd, 
            input,
            target,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
        //
    }

}
