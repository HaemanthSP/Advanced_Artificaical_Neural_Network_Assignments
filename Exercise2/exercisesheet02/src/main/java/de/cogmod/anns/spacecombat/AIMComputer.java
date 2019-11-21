package de.cogmod.anns.spacecombat;

import java.util.Random;

import de.cogmod.anns.math.Vector3d;

/**
 * @author Sebastian Otte
 */
public class AIMComputer implements SpaceSimulationObserver {

    private EnemySpaceShip enemy        = null;
    private boolean        targetlocked = false;
    
    private Vector3d[] enemytrajectoryprediction;
    
    
    public Vector3d[] getEnemyTrajectoryPrediction() {
        return this.enemytrajectoryprediction;
    }
    
    public boolean getTargetLocked() {
        return this.targetlocked;
    }
    
    public EnemySpaceShip getTarget() {
        return this.enemy;
    }
    
    public void releaseTarget() {
        synchronized (this) {
            this.enemy = null;
            this.targetlocked = false;
        }
    }
    
    public void lockTarget(final EnemySpaceShip enemy) {
        synchronized (this) {
            this.enemy        = enemy;
            this.targetlocked = true;
        }
    }
    
    // maybe only required for dummy trajectory.
    private Random rnd = new Random(1234);

    private Vector3d[] generateDummyFutureProjection(final int timesteps) {
        //
        Vector3d last           = this.enemy.getRelativePosition();
        final Vector3d dir      = new Vector3d();
        final Vector3d[] result = new Vector3d[timesteps]; 
        //
        for (int t = 0; t < timesteps; t++) {
            dir.x += rnd.nextGaussian() * 0.1;
            dir.y += rnd.nextGaussian() * 0.1;
            dir.z += rnd.nextGaussian() * 0.1;
            //
            Vector3d.normalize(dir, dir);
            //
            dir.x *= 1.0;
            dir.y *= 1.0;
            dir.z *= 1.0;
            
            final Vector3d current = Vector3d.add(last, dir);
            result[t] = Vector3d.add(current, enemy.getOrigin());
            last = current;
        }
        return result;
    }
    
    @Override
    public void simulationStep(final SpaceSimulation sim) {
        //
        synchronized (this) {
            //
            if (!this.targetlocked) return;
            
            //
            // update trajectory prediction RNN (teacher forcing)
            //
            final Vector3d enemyposition        = sim.getEnemy().getPosition();
            final Vector3d enemyrelativepostion = sim.getEnemy().getRelativePosition();
            // ...
            
            //
            // use copy of the RNN to generate future projection
            //
            this.enemytrajectoryprediction = this.generateDummyFutureProjection(100);
            
            //
            // will be later extended for missile control
            //
            
        }

    }
	
    
	
}