#include "CorrespondenceFilter.hpp"
#include "meshmonk/profiling.hpp"
#include <algorithm>
#include <limits>

namespace registration {

void CorrespondenceFilter::set_floating_input(
    const FeatureMat *const inFloatingFeatures,
    const VecDynFloat *const inFloatingFlags) {
  // # Set input
  _inFloatingFeatures = inFloatingFeatures;
  _inFloatingFlags = inFloatingFlags;

  // # Update internal parameters
  _numFloatingElements = _inFloatingFeatures->rows();
  _numAffinityElements = _numFloatingElements * _numNeighbours;

  // # Update the neighbour finder
  _neighbourFinder.set_queried_points(_inFloatingFeatures);
}

void CorrespondenceFilter::set_target_input(
    const FeatureMat *const inTargetFeatures,
    const VecDynFloat *const inTargetFlags) {
  // # Set input
  _inTargetFeatures = inTargetFeatures;
  _inTargetFlags = inTargetFlags;

  // # Update internal parameters
  _numTargetElements = _inTargetFeatures->rows();
  _numAffinityElements = _numFloatingElements * _numNeighbours;

  // # Update the neighbour finder
  _neighbourFinder.set_source_points(_inTargetFeatures);
}

void CorrespondenceFilter::set_parameters(const size_t numNeighbours,
                                          const float flagThreshold) {
  _numNeighbours = numNeighbours;
  _flagThreshold = flagThreshold;
  _numAffinityElements = _numFloatingElements * _numNeighbours;
  _neighbourFinder.set_parameters(_numNeighbours);
}

void CorrespondenceFilter::set_affinity_normalization(
    const bool normalizeAffinity) {
  _normalizeAffinity = normalizeAffinity;
}

void CorrespondenceFilter::set_target_faces(const FacesMat *const inTargetFaces) {
  _inTargetFaces = inTargetFaces;
  _build_vertex_face_adjacency();
}

void CorrespondenceFilter::_build_vertex_face_adjacency() {
  /*
  # GOAL
  Group target face ids by the vertices they touch (CSR layout), so that given
  a nearest vertex we can immediately gather the triangles around it as
  candidates for the closest-point projection.
  */
  _vertexFaceOffsets.clear();
  _vertexFaceIndices.clear();
  if (_inTargetFaces == NULL || _inTargetFeatures == NULL) {
    return;
  }

  const size_t numVertices = _inTargetFeatures->rows();
  const size_t numFaces = _inTargetFaces->rows();

  // # Count incident faces per vertex, then prefix-sum into offsets.
  std::vector<int> counts(numVertices, 0);
  for (size_t f = 0; f < numFaces; f++) {
    for (int c = 0; c < 3; c++) {
      const int v = (*_inTargetFaces)(f, c);
      if (v >= 0 && v < (int)numVertices) {
        counts[v]++;
      }
    }
  }
  _vertexFaceOffsets.resize(numVertices + 1, 0);
  for (size_t v = 0; v < numVertices; v++) {
    _vertexFaceOffsets[v + 1] = _vertexFaceOffsets[v] + counts[v];
  }
  _vertexFaceIndices.resize(_vertexFaceOffsets[numVertices]);

  // # Fill, reusing counts as a per-vertex write cursor.
  std::vector<int> cursor(_vertexFaceOffsets.begin(),
                          _vertexFaceOffsets.end() - 1);
  for (size_t f = 0; f < numFaces; f++) {
    for (int c = 0; c < 3; c++) {
      const int v = (*_inTargetFaces)(f, c);
      if (v >= 0 && v < (int)numVertices) {
        _vertexFaceIndices[cursor[v]++] = (int)f;
      }
    }
  }
}

namespace {

// Closest point to `p` on triangle (a,b,c), after Ericson, Real-Time Collision
// Detection §5.1.5. Also returns the barycentric coordinates so per-vertex
// attributes (normals, flags) can be interpolated at that point.
inline Vec3Float closest_point_on_triangle(const Vec3Float &p,
                                           const Vec3Float &a,
                                           const Vec3Float &b,
                                           const Vec3Float &c, float &outU,
                                           float &outV, float &outW) {
  const Vec3Float ab = b - a;
  const Vec3Float ac = c - a;
  const Vec3Float ap = p - a;
  const float d1 = ab.dot(ap);
  const float d2 = ac.dot(ap);
  if (d1 <= 0.0f && d2 <= 0.0f) {
    outU = 1.0f; outV = 0.0f; outW = 0.0f;
    return a;
  }

  const Vec3Float bp = p - b;
  const float d3 = ab.dot(bp);
  const float d4 = ac.dot(bp);
  if (d3 >= 0.0f && d4 <= d3) {
    outU = 0.0f; outV = 1.0f; outW = 0.0f;
    return b;
  }

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    outU = 1.0f - v; outV = v; outW = 0.0f;
    return a + v * ab;
  }

  const Vec3Float cp = p - c;
  const float d5 = ab.dot(cp);
  const float d6 = ac.dot(cp);
  if (d6 >= 0.0f && d5 <= d6) {
    outU = 0.0f; outV = 0.0f; outW = 1.0f;
    return c;
  }

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / (d2 - d6);
    outU = 1.0f - w; outV = 0.0f; outW = w;
    return a + w * ac;
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    outU = 0.0f; outV = 1.0f - w; outW = w;
    return b + w * (c - b);
  }

  const float denom = 1.0f / (va + vb + vc);
  const float v = vb * denom;
  const float w = vc * denom;
  outU = 1.0f - v - w; outV = v; outW = w;
  return a + ab * v + ac * w;
}

} // namespace

void CorrespondenceFilter::_update_point_to_surface_correspondences() {
  /*
  # GOAL
  Give each floating vertex the closest point on the target *surface* instead
  of a distance-weighted blend of nearby target vertices.

  A blend of vertices depends on how many vertices happen to sit nearby, so
  retessellating the same surface moves the correspondence. Projecting onto the
  triangles removes that: the triangles describe the same surface no matter how
  finely it is subdivided.

  Candidate triangles are those incident to the k nearest target vertices,
  which is where the true closest point lies except in pathological cases.

  The result is written twice, to the corresponding features directly and to
  the affinity matrix as barycentric weights. The direct write is what the
  non-symmetric path consumes. The affinity rebuild is what makes this usable
  from SymmetricCorrespondenceFilter, which fuses its push and pull filters
  through get_affinity() and never looks at their features -- so without it,
  a symmetric registration would compute point-to-surface and discard it.

  The two agree by construction: a closest point on a triangle *is* the
  barycentric combination u*a + v*b + w*c of the triangle's vertices, so a
  three-entry affinity row reproduces it exactly, normals included, since
  features carry position and normal in the same row.
  */
  const MatDynInt neighbourIndices = _neighbourFinder.get_indices();
  const size_t numTargetVertices = _inTargetFeatures->rows();

  // Rows that found no triangle keep their blended affinity, so seed the
  // triplet list from the existing matrix and overwrite only what is resolved.
  std::vector<Triplet> affinityElements;
  affinityElements.reserve(3 * _numFloatingElements);
  std::vector<bool> resolved(_numFloatingElements, false);

  for (size_t i = 0; i < _numFloatingElements; i++) {
    const Vec3Float p = _inFloatingFeatures->row(i).head(3);

    float bestDistSq = std::numeric_limits<float>::max();
    Vec3Float bestPoint = p;
    int bestFace = -1;
    float bestU = 1.0f, bestV = 0.0f, bestW = 0.0f;

    for (size_t j = 0; j < _numNeighbours; j++) {
      const int nb = neighbourIndices(i, j);
      if (nb < 0 || nb >= (int)numTargetVertices) {
        continue;
      }
      const int begin = _vertexFaceOffsets[nb];
      const int end = _vertexFaceOffsets[nb + 1];
      for (int fi = begin; fi < end; fi++) {
        const int f = _vertexFaceIndices[fi];
        const int i0 = (*_inTargetFaces)(f, 0);
        const int i1 = (*_inTargetFaces)(f, 1);
        const int i2 = (*_inTargetFaces)(f, 2);
        const Vec3Float a = _inTargetFeatures->row(i0).head(3);
        const Vec3Float b = _inTargetFeatures->row(i1).head(3);
        const Vec3Float c = _inTargetFeatures->row(i2).head(3);
        float u, v, w;
        const Vec3Float q = closest_point_on_triangle(p, a, b, c, u, v, w);
        const float distSq = (q - p).squaredNorm();
        if (distSq < bestDistSq) {
          bestDistSq = distSq;
          bestPoint = q;
          bestFace = f;
          bestU = u; bestV = v; bestW = w;
        }
      }
    }

    if (bestFace < 0) {
      // # No incident triangles (isolated vertices): fall back to the blended
      // # correspondence already computed from the affinity matrix.
      continue;
    }

    const int i0 = (*_inTargetFaces)(bestFace, 0);
    const int i1 = (*_inTargetFaces)(bestFace, 1);
    const int i2 = (*_inTargetFaces)(bestFace, 2);

    // # Features are only written when a consumer attached an output buffer.
    // # The symmetric filter's sub-filters have none and read the affinity.
    if (_ioCorrespondingFeatures != NULL) {
      // # Interpolate the normal at the projected point.
      Vec3Float n = bestU * Vec3Float(_inTargetFeatures->row(i0).tail(3)) +
                    bestV * Vec3Float(_inTargetFeatures->row(i1).tail(3)) +
                    bestW * Vec3Float(_inTargetFeatures->row(i2).tail(3));
      const float nNorm = n.norm();
      if (nNorm > 1e-8f) {
        n /= nNorm;
      }

      (*_ioCorrespondingFeatures)(i, 0) = bestPoint[0];
      (*_ioCorrespondingFeatures)(i, 1) = bestPoint[1];
      (*_ioCorrespondingFeatures)(i, 2) = bestPoint[2];
      (*_ioCorrespondingFeatures)(i, 3) = n[0];
      (*_ioCorrespondingFeatures)(i, 4) = n[1];
      (*_ioCorrespondingFeatures)(i, 5) = n[2];

      // # A correspondence is only as trustworthy as the triangle it landed
      // # on, so require all three of its vertices to be flagged valid.
      const float flag = std::min({(*_inTargetFlags)[i0], (*_inTargetFlags)[i1],
                                   (*_inTargetFlags)[i2]});
      (*_ioCorrespondingFlags)[i] = (flag > _flagThreshold) ? 1.0f : 0.0f;
    }

    // # Same correspondence expressed as affinity, for the symmetric filter.
    affinityElements.push_back(Triplet(i, i0, bestU));
    affinityElements.push_back(Triplet(i, i1, bestV));
    affinityElements.push_back(Triplet(i, i2, bestW));
    resolved[i] = true;
  }

  // # Carry over the blended rows for floating vertices that found no triangle.
  for (int k = 0; k < _affinity.outerSize(); k++) {
    for (SparseMat::InnerIterator it(_affinity, k); it; ++it) {
      if (!resolved[it.row()]) {
        affinityElements.push_back(Triplet(it.row(), it.col(), it.value()));
      }
    }
  }
  _affinity.setZero();
  _affinity.setFromTriplets(affinityElements.begin(), affinityElements.end());
}

void CorrespondenceFilter::_update_affinity() {
  /*
  # GOALthe
  For each element in _inFloatingFeatures, we're going to determine affinity
  weights which link it to elements in _inTargetFeatures. This is based on the
  Euclidean distance of k nearest neighbours found for each element of
  _inFloatingFeatures.
  */

  // # Initialization
  // ## Initialize the sparse affinity matrix
  _affinity = SparseMat(_numFloatingElements, _numTargetElements);
  _affinity.reserve(_numAffinityElements);
  // ## Initialize a vector of triplets which is used to insert elements into
  // ## the affinity matrix later.
  std::vector<Triplet> affinityElements(_numAffinityElements,
                                        Triplet(0, 0, 0.0f));

  // ## Obtain pointers to neighbouring indices and (squared) distances:
  const MatDynInt neighbourIndices = _neighbourFinder.get_indices();
  const MatDynFloat neighbourSquaredDistances =
      _neighbourFinder.get_distances();

  // # Compute the affinity matrix
  // ## Loop over the first feature set to determine their affinity with the
  // ## second set.
  size_t i = 0;
  size_t j = 0;
  Vec3Float floatingNormal = Vec3Float::Zero();
  Vec3Float targetNormal = Vec3Float::Zero();
  unsigned int counter = 0;
  for (; i < _numFloatingElements; i++) {
    floatingNormal = _inFloatingFeatures->row(i).tail(3);
    // ### Loop over each found neighbour
    for (j = 0; j < _numNeighbours; j++) {
      // ### Get index of neighbour and squared distance to it
      const int neighbourIndex = neighbourIndices(i, j);
      float distanceSquared = neighbourSquaredDistances(i, j);

      // ### For numerical stability, check if the distance is very small
      const float eps1 = 0.000001f;
      if (distanceSquared < eps1) {
        distanceSquared = eps1;
      }

      // ### Compute the affinity element as 1/distance*distance
      float affinityElement = 1.0f / distanceSquared;

      // ### Incorporate the orientation
      targetNormal = _inTargetFeatures->row(neighbourIndex).tail(3);
      float dotProduct = floatingNormal.dot(targetNormal);
      float orientationWeight = dotProduct / 2.0f + 0.5f;
      affinityElement *= orientationWeight;

      // ### Incorporate the surface area this neighbour represents, so that a
      // ### finely tessellated patch does not outvote a coarsely tessellated
      // ### one covering the same amount of surface.
      if (_inSourceAreas != NULL &&
          neighbourIndex < (int)_inSourceAreas->size()) {
        affinityElement *= (*_inSourceAreas)[neighbourIndex];
      }

      // ### Check for numerical stability (the affinity elements will be
      // ### normalized later, so dividing by a sum of tiny elements might
      // ### go wrong.
      const float eps2 = 0.0001f;
      if (affinityElement < eps2) {
        affinityElement = eps2;
      }

      // ### Write result to the affinity triplet list
      affinityElements[counter] = Triplet(i, neighbourIndex, affinityElement);
      counter++;
    }
  }
  // ## Construct the sparse matrix with the computed element list
  _affinity.setFromTriplets(affinityElements.begin(), affinityElements.end());

  // # Normalize the rows of the affinity matrix
  if (_normalizeAffinity) {
    normalize_sparse_matrix(_affinity);
  }

} // end wknn_affinity()

void CorrespondenceFilter::update() {
#ifdef MESHMONK_PROFILING
  auto _t = g_profiler.scoped("CorrespondenceFilter::update");
#endif

  // # Update the neighbour indices and distances
  _neighbourFinder.set_queried_points(_inFloatingFeatures);
  _neighbourFinder.update();

  // # Update the (sparse) affinity matrix
  _update_affinity();

  if (_ioCorrespondingFeatures != NULL) {
    // # Use the affinity weights to determine corresponding features and flags.
    if (_normalizeAffinity) {
      BaseCorrespondenceFilter::_affinity_to_correspondences();
    } else {
      // ## You shouldn't compute correspondences without normalization of the
      // affinity matrix. However the
      // ## user requested no normalization of the affinity matrix. We therefor
      // temporarily normalize
      // ## the affinity matrix to compute correspondences, and then restore the
      // non-normalized affinity
      // ## matrix.

      // ### Copy non-normalized affinity matrix
      SparseMat affinityCopy = _affinity;

      // ### Normalize the affinity matrix
      normalize_sparse_matrix(_affinity);

      // ### Compute corresponding features and flags
      BaseCorrespondenceFilter::_affinity_to_correspondences();

      // ### Restore the affinity matrix
      _affinity = affinityCopy;
    }
  } else {
    /*
    affinity is computed and get be requested from the filter, but output
    correspondences are not generated if the filter doesn't know where to write
    the output.
    */
  }

  // # Refine the blended correspondences into closest points on the target
  // # surface, which does not depend on how the target was tessellated.
  //
  // # This runs outside the block above because it also rewrites the affinity,
  // # and SymmetricCorrespondenceFilter drives its push and pull sub-filters
  // # with no output attached -- it consumes get_affinity() alone. Leaving this
  // # inside would silently skip point-to-surface for symmetric registrations.
  if (_inTargetFaces != NULL && !_vertexFaceOffsets.empty()) {
    _update_point_to_surface_correspondences();
  }
} // end wkkn_correspondences()

} // namespace registration
