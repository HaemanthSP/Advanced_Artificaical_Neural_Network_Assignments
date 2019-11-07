package de.cogmod.anns.exercisesheet01.misc;

/**
 * @author Sebastian Otte
 */
public interface LearningListener {
    public void afterEpoch(final int epoch, final double trainingerror);
}