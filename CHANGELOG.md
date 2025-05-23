
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

### Changed
- R2D2 and ECBC are only non-stationary methods.

### Removed
- AR2D2 and QMrs are deprecated, use R2D2.

### Fixed

