#include "NonrigidRegistration.hpp"
#include "meshmonk/profiling.hpp"
#include <cstdlib>
#include <memory>

namespace registration {

void NonrigidRegistration::set_input(FeatureMat *const ioFloatingFeatures,
                                     const FeatureMat *const inTargetFeatures,
                                     const FacesMat *const inFloatingFaces,
                                     const VecDynFloat *const inFloatingFlags,
                                     const VecDynFloat *const inTargetFlags) {
  _ioFloatingFeatures = ioFloatingFeatures;
  _inTargetFeatures = inTargetFeatures;
  _inFloatingFaces = inFloatingFaces;
  _inFloatingFlags = inFloatingFlags;
  _inTargetFlags = inTargetFlags;
} // end set_input()

void NonrigidRegistration::set_parameters(
    bool symmetric, size_t numNeighbours, float flagThreshold,
    bool equalizePushPull, float kappaa, bool inlierUseOrientation,
    size_t numIterations, float sigmaSmoothing,
    size_t numViscousIterationsStart, size_t numViscousIterationsEnd,
    size_t numElasticIterationsStart, size_t numElasticIterationsEnd) {
  _symmetric = symmetric;
  _numNeighbours = numNeighbours;
  _flagThreshold = flagThreshold;
  _equalizePushPull = equalizePushPull;
  _kappaa = kappaa;
  _inlierUseOrientation = inlierUseOrientation;
  _numIterations = numIterations;
  _sigmaSmoothing = sigmaSmoothing;
  _numViscousIterationsStart = numViscousIterationsStart;
  _numViscousIterationsEnd = numViscousIterationsEnd;
  _numElasticIterationsStart = numElasticIterationsStart;
  _numElasticIterationsEnd = numElasticIterationsEnd;
  _numViscousIterations = _numViscousIterationsStart;
  _numElasticIterations = _numElasticIterationsStart;

  if (_numViscousIterationsStart > 0) {
    _viscousAnnealingRate = exp(log(float(_numViscousIterationsEnd) /
                                    float(_numViscousIterationsStart)) /
                                (_numIterations - 1));
  } else {
    _viscousAnnealingRate = 1.0f;
  }
  if (_numElasticIterationsStart > 0) {
    _elasticAnnealingRate = exp(log(float(_numElasticIterationsEnd) /
                                    float(_numElasticIterationsStart)) /
                                (_numIterations - 1));
  } else {
    _elasticAnnealingRate = 1.0f;
  }
} // end set_parameters()

void NonrigidRegistration::update() {
#ifdef MESHMONK_PROFILING
  auto _t = g_profiler.scoped("NonrigidRegistration::update");
#endif

  // # Initializes
  size_t numFloatingVertices = _ioFloatingFeatures->rows();
  FeatureMat correspondingFeatures =
      FeatureMat::Zero(numFloatingVertices, registration::NUM_FEATURES);
  VecDynFloat correspondingFlags = VecDynFloat::Zero(numFloatingVertices);

  // # Per-vertex surface areas, so correspondences weight surface rather than
  // # vertex count (see compute_vertex_areas). Both meshes keep their
  // # connectivity throughout, so these are computed once up front; the
  // # floating ones are taken from its starting pose.
  VecDynFloat floatingAreas;
  VecDynFloat targetAreas;
  // Off by default on this branch so point-to-surface is measured on its own,
  // but switchable at runtime because the two are complementary rather than
  // alternatives: point-to-surface makes the *push* direction independent of
  // target sampling, while the *pull* still sums one contribution per target
  // vertex -- so a finer tessellation pulls harder for no physical reason.
  // Area weighting is what fixes that second half. Comparing them needs the
  // same binary, so this is an env switch rather than a rebuild.
  const char *areaEnv = std::getenv("MESHMONK_AREA_WEIGHTING");
  const bool useAreaWeighting = (areaEnv != NULL && areaEnv[0] == '1') &&
                                (_inTargetFaces != NULL) &&
                                (_inFloatingFaces != NULL);
  if (useAreaWeighting) {
    floatingAreas = compute_vertex_areas(*_ioFloatingFeatures, *_inFloatingFaces);
    targetAreas = compute_vertex_areas(*_inTargetFeatures, *_inTargetFaces);
  }

  // # Set up the filters
  // ## Correspondence Filter
  std::unique_ptr<BaseCorrespondenceFilter> correspondenceFilter;
  // Non-owning aliases; exactly one is set, depending on the mode. Both filters
  // can take connectivity, so point-to-surface is available either way.
  CorrespondenceFilter *plainFilter = NULL;
  SymmetricCorrespondenceFilter *symmetricFilter = NULL;

  if (_symmetric) {
    auto *f = new SymmetricCorrespondenceFilter();
    f->set_parameters(_numNeighbours, _flagThreshold, _equalizePushPull);
    if (useAreaWeighting) {
      f->set_areas(&floatingAreas, &targetAreas);
    }
    symmetricFilter = f;
    correspondenceFilter.reset(f);
  } else {
    auto *f = new CorrespondenceFilter();
    f->set_parameters(_numNeighbours, _flagThreshold);
    if (useAreaWeighting) {
      f->set_source_areas(&targetAreas);
    }
    plainFilter = f;
    correspondenceFilter.reset(f);
  }
  correspondenceFilter->set_floating_input(_ioFloatingFeatures,
                                           _inFloatingFlags);
  correspondenceFilter->set_target_input(_inTargetFeatures, _inTargetFlags);
  correspondenceFilter->set_output(&correspondingFeatures, &correspondingFlags);

  // # Connectivity enables point-to-surface correspondences. It has to come
  // # after set_target_input(), which is what the adjacency is built over.
  //
  // # The symmetric filter takes both meshes: its push direction projects the
  // # template onto the target surface, its pull direction does the reverse.
  // # Keeping the pull direction is what stops the template sliding off thin,
  // # high-curvature structures -- on LAFAS, dropping it moved the ear
  // # landmarks by up to 9mm while the rest of the face was unaffected.
  if (plainFilter != NULL && _inTargetFaces != NULL) {
    plainFilter->set_target_faces(_inTargetFaces);
  }
  if (symmetricFilter != NULL) {
    symmetricFilter->set_faces(_inFloatingFaces, _inTargetFaces);
  }

  // ## Inlier Filter
  VecDynFloat floatingWeights = VecDynFloat::Ones(numFloatingVertices);
  InlierDetector inlierDetector;
  inlierDetector.set_input(_ioFloatingFeatures, &correspondingFeatures,
                           &correspondingFlags);
  inlierDetector.set_output(&floatingWeights);
  inlierDetector.set_parameters(_kappaa, _inlierUseOrientation);
  // ## Transformation Filter
  _numViscousIterations = _numViscousIterationsStart;
  _numElasticIterations = _numElasticIterationsStart;
  ViscoElasticTransformer transformer;
  transformer.set_input(&correspondingFeatures, &floatingWeights,
                        _inFloatingFlags, _inFloatingFaces);
  transformer.set_output(_ioFloatingFeatures);

  // # Perform ICP
  for (size_t iteration = 0; iteration < _numIterations; iteration++) {

    // # Anneal parameters
    _numViscousIterations =
        int(std::round(_numViscousIterationsStart *
                       std::pow(_viscousAnnealingRate, iteration)));
    if (_numViscousIterations < _numViscousIterationsEnd) {
      _numViscousIterations = _numViscousIterationsEnd;
    }
    _numElasticIterations =
        int(std::round(_numElasticIterationsStart *
                       std::pow(_elasticAnnealingRate, iteration)));
    if (_numElasticIterations < _numElasticIterationsEnd) {
      _numElasticIterations = _numElasticIterationsEnd;
    }

    // # Correspondences
    correspondenceFilter->set_floating_input(_ioFloatingFeatures,
                                             _inFloatingFlags);
    correspondenceFilter->set_target_input(_inTargetFeatures, _inTargetFlags);
    correspondenceFilter->update();

    // # Inlier Detection
    inlierDetector.update();

    // # Transformation
    transformer.set_parameters(10, _sigmaSmoothing, _numViscousIterations,
                               _numElasticIterations);
    transformer.update();
  }

  // correspondenceFilter is automatically deleted by unique_ptr

} // end update()

} // namespace registration
