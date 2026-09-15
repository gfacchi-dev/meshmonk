#include "SymmetricCorrespondenceFilter.hpp"
#include "meshmonk/profiling.hpp"

namespace registration {

void SymmetricCorrespondenceFilter::set_floating_input(
    const FeatureMat *const inFloatingFeatures,
    const VecDynFloat *const inFloatingFlags) {
  // # Set input
  _inFloatingFeatures = inFloatingFeatures;
  _inFloatingFlags = inFloatingFlags;

  // # Update internal parameters
  _numFloatingElements = _inFloatingFeatures->rows();

  // # Update the push and pull filters
  _pushFilter.set_floating_input(_inFloatingFeatures, _inFloatingFlags);
  _pullFilter.set_target_input(_inFloatingFeatures, _inFloatingFlags);
}

void SymmetricCorrespondenceFilter::set_target_input(
    const FeatureMat *const inTargetFeatures,
    const VecDynFloat *const inTargetFlags) {
  // # Set input
  _inTargetFeatures = inTargetFeatures;
  _inTargetFlags = inTargetFlags;

  // # Update internal parameters
  _numTargetElements = _inTargetFeatures->rows();

  // # Update the push and pull filters
  _pushFilter.set_target_input(_inTargetFeatures, _inTargetFlags);
  _pullFilter.set_floating_input(_inTargetFeatures, _inTargetFlags);
}

void SymmetricCorrespondenceFilter::set_parameters(
    const size_t numNeighbours, const float flagThreshold,
    const bool equalizePushPull) {
  _numNeighbours = numNeighbours;
  _flagThreshold = flagThreshold;
  _equalizePushPull = equalizePushPull;
  _pushFilter.set_parameters(_numNeighbours, _flagThreshold);
  _pullFilter.set_parameters(_numNeighbours, _flagThreshold);
  _pushFilter.set_affinity_normalization(false);
  _pullFilter.set_affinity_normalization(false);
}

void SymmetricCorrespondenceFilter::_update_push_and_pull() {

  // # Compute the push and pull affinity
  _pushFilter.update();
  _pullFilter.update();
  // # Get the affinities
  _affinity = _pushFilter.get_affinity();
  SparseMat pullAffinity = _pullFilter.get_affinity();

  // # Normalize the affinities before fusing them?
  if (_equalizePushPull) {
    normalize_sparse_matrix(_affinity);
    normalize_sparse_matrix(pullAffinity);
  }

  // # Weight each target vertex's pull by the surface area it represents.
  /*
  Rows of pullAffinity are target vertices, and fusing sums them into the
  floating rows -- so without this, the pull a floating vertex feels grows with
  the *number* of nearby target vertices rather than the amount of surface
  they cover. Two scans of the same face with different tessellation then
  converge to different results. Scaling by area makes the fused pull a surface
  integral instead of a vertex count.

  This has to happen after the normalization above: normalizing rows first and
  then scaling preserves the area weighting, whereas scaling first would have
  it divided straight back out.
  */
  if (_inTargetAreas != NULL) {
    scale_sparse_matrix_rows(pullAffinity, *_inTargetAreas);
  }

  // # Fuse the affinities
  fuse_affinities(_affinity,
                  pullAffinity); // helper function to combine affinity matrices

} // end wknn_affinity()

void SymmetricCorrespondenceFilter::update() {
#ifdef MESHMONK_PROFILING
  auto _t = g_profiler.scoped("SymmetricCorrespondenceFilter::update");
#endif
  // # Update the I/O for the push- and pull-filters
  _pushFilter.set_floating_input(_inFloatingFeatures, _inFloatingFlags);
  _pushFilter.set_target_input(_inTargetFeatures, _inTargetFlags);
  _pullFilter.set_floating_input(_inTargetFeatures, _inTargetFlags);
  _pullFilter.set_target_input(_inFloatingFeatures, _inFloatingFlags);

  // # Each filter searches over the point set it was given as *target*, so the
  // # areas it should weight by are that mesh's: the target mesh for push, the
  // # floating mesh for pull (their roles are swapped).
  _pushFilter.set_source_areas(_inTargetAreas);
  _pullFilter.set_source_areas(_inFloatingAreas);

  // # Set the output for the pull filter so that we can extract the pull flags
  // later.
  FeatureMat pullFeatures = FeatureMat::Zero(_numTargetElements, NUM_FEATURES);
  VecDynFloat pullFlags = VecDynFloat::Ones(_numTargetElements);
  _pullFilter.set_output(&pullFeatures, &pullFlags);

  // # Update the push and pull filters.
  // ## This gives us the affinity matrices for both push and pull directions,
  // ## and the pull flags.
  _update_push_and_pull();

  // # The pull flags are updated. We now compute the corresponding pull flags.
  VecDynFloat correspondingPullFlags = VecDynFloat::Ones(_numFloatingElements);
  correspondingPullFlags = _affinity * pullFlags;

  // # Use the affinity weights to determine corresponding features and flags.
  BaseCorrespondenceFilter::_affinity_to_correspondences();

  // # Merge the just computed corresponding flags with the corresponding pull
  // flags.
  (*BaseCorrespondenceFilter::_ioCorrespondingFlags) =
      (*BaseCorrespondenceFilter::_ioCorrespondingFlags)
          .cwiseProduct(correspondingPullFlags);

  // # Make these final flags binary again
  for (size_t i = 0; i < _numFloatingElements; i++) {
    if ((*BaseCorrespondenceFilter::_ioCorrespondingFlags)[i] >
        BaseCorrespondenceFilter::_flagThreshold) {
      (*BaseCorrespondenceFilter::_ioCorrespondingFlags)[i] = 1.0f;
    } else {
      (*BaseCorrespondenceFilter::_ioCorrespondingFlags)[i] = 0.0f;
    }
  }

} // end wkkn_correspondences()

} // namespace registration
