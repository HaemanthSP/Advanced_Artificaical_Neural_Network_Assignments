package de.cogmod.anns.spacecombat.rnn;

import java.util.Arrays;
import java.util.Random;

import de.cogmod.anns.spacecombat.EnemySpaceShip;
import de.cogmod.anns.math.Vector3d;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;
import de.jannlab.optimization.optimizer.DifferentialEvolution.Mutation;
import static de.cogmod.anns.spacecombat.rnn.ReservoirTools.*;


public class EchoStateNetworkMain {
    static double[] vector_to_array(Vector3d v) {
        return new double[]{v.x, v.y, v.z};
    }

    public static void main(String[] args) {

        EnemySpaceShip train_enemy = new EnemySpaceShip();
        train_enemy.reset(new Random(1));
        final double[][] train_washout = new double[100][];
        final double[][] train_sequence = new double[400][];
        //System.out.println("import matplotlib.pyplot as plt");
        //System.out.println("traj = [");
        for (int t = 0; t < train_washout.length; ++t) {
            train_enemy.update();
            train_washout[t] = vector_to_array(train_enemy.getRelativePosition());
            //System.out.printf("%s,\n", Arrays.toString(train_washout[t]));
        }
        //System.out.println("]");
        //System.out.println("t = [i[0] for i in traj]; plt.plot(t); plt.show()");
        //System.out.println("t = [i[1] for i in traj]; plt.plot(t); plt.show()");
        //System.out.println("t = [i[2] for i in traj]; plt.plot(t); plt.show()");

        //assert(false);
        for (int t = 0; t < train_sequence.length; ++t) {
            train_enemy.update();
            train_sequence[t] = vector_to_array(train_enemy.getRelativePosition());
        }

        EnemySpaceShip eval_enemy = new EnemySpaceShip();
        eval_enemy.reset(new Random(2));
        final double[][] eval_washout = new double[100][];
        final double[][] eval_sequence = new double[1000][];
        for (int t = 0; t < eval_washout.length; ++t) {
            eval_enemy.update();
            eval_washout[t] = vector_to_array(eval_enemy.getRelativePosition());
        }
        for (int t = 0; t < eval_sequence.length; ++t) {
            eval_enemy.update();
            eval_sequence[t] = vector_to_array(eval_enemy.getRelativePosition());
        }

        final EchoStateNetwork esn = new EchoStateNetwork(3, 50, 3);

        // this effectively only affects output feedback weights:
        // recurrent weights are initialized within the differential
        // evolution algorithm and readout weights are optimized
        // with least squares pseudoinverse computation
        double feedback_scale = 3e-7;
        esn.initializeWeights(new Random(1234), feedback_scale);

        final Objective f = new Objective() {
            @Override
            public int arity() {
                return esn.reservoirsize * esn.reservoirsize;
            }

            @Override
            public double compute(double[] values, int offset) {
                map(values, offset, esn.getReservoirWeights());
                return esn.trainESN(train_washout, train_sequence);
            }
        };

        final DifferentialEvolution optimizer = new DifferentialEvolution();
        optimizer.setF(0.4);
        optimizer.setCR(0.6);
        optimizer.setPopulationSize(10);
        optimizer.setMutation(Mutation.CURR2RANDBEST_ONE);
        optimizer.setInitLbd(-0.1);
        optimizer.setInitUbd(0.1);
        optimizer.setRnd(new Random(1234));
        optimizer.setParameters(f.arity());
        optimizer.updateObjective(f);
        optimizer.addListener(new BasicOptimizationListener());
        optimizer.initialize();
        optimizer.iterate(500, 0.0);

        final double[] solution = new double[f.arity()];
        optimizer.readBestSolution(solution, 0);
        map(solution, 0, esn.getReservoirWeights());
        esn.trainESN(train_washout, train_sequence);

        //double train_mse = esn.evaluateESN(train_washout, train_sequence, Double.toString(feedback_scale) + "_train");
        //double eval_mse = esn.evaluateESN(eval_washout, eval_sequence, Double.toString(feedback_scale) + "_eval");
        //System.out.printf("open('%s.txt', 'w').write('train mse = %.3f\\neval mse = %.3f')\n", Double.toString(feedback_scale), train_mse, eval_mse);
        //System.out.printf("print('train mse = %.3f')", train_mse);
        //System.out.printf("print('eval mse = %.3f')", eval_mse);

        esn.printWeights();
    }
}
