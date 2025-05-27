
# Changelog

## [Unreleased]

### Added
- AbstractBC: a base class for all BC methods
- UnivariateBC: a base class for univariate BC methods
- MultiUBC: a base class to transform a multivariate method into multivariate
  (but independent) method.
- Decorators: io_fit and io_predict. use to transform input / output in 2d
- CDFt: many configurations added
- dOTC1d: new class for dOTC in 1d, solved with quantile mapping (faster than
  simplex)
- QQD: Quantile-Quantile of Dequé.
- New SBCK.stats modules:
   - Move all elements of SBCK.metrics to SBCK.stats.__sparse_distance, and add
     typing and some corrections.

### Changed
- R2D2 and ECBC are only non-stationary methods.
- Add normalization in distance SBCK.stats.chebyshev

### Removed
- AR2D2 and QMrs are deprecated, use R2D2.
- Remove SBCK.stats.entropy
- Remove the module SBCK.metrics (moved to SBCK.stats.__sparse_distance)

### Fixed

