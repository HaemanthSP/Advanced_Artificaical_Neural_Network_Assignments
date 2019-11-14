package de.cogmod.anns.exercisesheet01;

import java.util.Random;

import de.cogmod.anns.exercisesheet01.RecurrentNeuralNetwork;
import de.cogmod.anns.exercisesheet01.misc.BasicLearningListener;

/**
 * @author Franciszek Piszcz, Haemanth Santhi Ponnusamy
 */
public class GradientMeasurement {
    public static void main(String[] args) {
        final double[][] input = {
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
          {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
        };
        final Random rnd = new Random(100L);
        final RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, 1, 1);
        // we disable all biases
        net.setBias(1, false);
        net.setBias(2, false);
        net.initializeWeights(rnd, 0.1);
        net.rebufferOnDemand(100);
        final double[][] target = net.forwardPass(input);
        System.out.println(target.length);
        target[99][0] = 1.0;
        net.backwardPass(target);
        net.printDeltas();
    }
}
